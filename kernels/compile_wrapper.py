from __future__ import annotations

import torch

from inspect import signature
from operator import attrgetter
from types import NoneType
from typing import Any, Callable, Dict, Annotated, List, TypeVar, Union, Tuple

from msgspec import Struct


__all__ = ['Dynamic', 'BaseKernel']


T = TypeVar('T')


NAMESPACE: str = "teal"

TENSOR: str = "Tensor"
NONE: str = "None"

DOT_OP: str = "."
TYPE_SEP: str = " | "
COMMA_SPACE: str = ", "

TYPE_MAPPINGS: Dict[str, str] = (
    {
        'dtype': 'ScalarType',
        'device': 'Device',
    }
)


class Dynamic:

    Int = Annotated[int, 'SymInt']
    Num = Annotated[int | float, 'Scalar']


def check_origin(annotation: Any, target: Any) -> bool:

    return hasattr(annotation, '__origin__') and annotation.__origin__ is target


def unpack_optional(annotation: Union[T, NoneType] | T) -> Tuple[T, bool]:

    if check_origin(annotation, Union):

        target, maybe_none, *maybe_more = annotation.__args__

        if maybe_more or maybe_none is not NoneType:

            raise TypeError(f'unsupported multi-union type: {annotation}')

        else:

            return target, True

    return annotation, False


def unpack_list(annotation: List[T] | T) -> Tuple[T, bool]:

    if check_origin(annotation, list):

        return annotation.__args__[0], True

    return annotation, False


def resolve_type(annotation: Any) -> str:

    target, is_optional = unpack_optional(annotation)
    target, is_iterable = unpack_list(target)

    if hasattr(target, '__metadata__'):

        resolved = target.__metadata__[0]

    elif hasattr(target, '__name__'):

        if (str_value := target.__name__).endswith(TENSOR):

            resolved = TENSOR

        else:

            resolved = TYPE_MAPPINGS.get(str_value, str_value)

    else:

        raise TypeError(f'unable to infer type from given {target}')

    if is_iterable:

        resolved = f'{resolved}[]'

    if is_optional:

        resolved = f'{resolved}?'

    return resolved


def resolve_return(annotation: Any) -> str:

    target = annotation

    if check_origin(annotation, tuple):

        target = annotation.__args__

    if isinstance(target, tuple):

        return f'({COMMA_SPACE.join(map(resolve_type, target))})'

    else:

        return resolve_type(target)


class BaseKernel(Struct):

    """
    Base helper dataclass for wrapping custom kernels for registration with `torch.library`.

    Note(s):

        • The underlying `schema` is automatically inferred from the type annotation of the
          `forward` method.

        • Should this fail to recognize non-Tensor(s), feel free to override the `schematize`
          class-method.

    """

    # Library name and corresponding target device.
    name: str
    target: str

    # The auto-generated signature.
    schema: str

    @classmethod
    def initialize(cls, name: str, target: str, **kwargs) -> BaseKernel:

        return cls(name, target, cls.schematize())

    @classmethod
    def schematize(cls) -> str:

        params = dict((forward_signature := signature(cls.forward, eval_str=True)).parameters)

        _ = params.pop('self')

        arguments = (f"{resolve_type(p.annotation)} {name}" for name, p in params.items())

        out = resolve_return(forward_signature.return_annotation)

        return f"({', '.join(arguments)}) -> {out}"

    @property
    def is_registered(self) -> bool:

        return hasattr(getattr(torch.ops, NAMESPACE), self.name)

    def operator(self, compiled: bool = False) -> Callable:

        if compiled:

            self.register()

            return attrgetter(f'{NAMESPACE}.{self.name}')(torch.ops)

        else:

            return self.forward

    def meta(self, *args, **kwargs) -> Any:

        raise NotImplementedError(
            f'abstract implementation `meta` in {self.__class__.__name__} required for '
            f'registration'
        )

    def forward(self, *args, **kwargs) -> Any:

        raise NotImplementedError(
            f'concrete implementation `forward` in {self.__class__.__name__} required for '
            f'registration'
        )

    def register(self) -> None:

        if not self.is_registered:

            qualname = f'{NAMESPACE}::{self.name}'

            # Define the library namespace and signature.
            torch.library.define(qualname, self.schema)

            # Register the abstract implementation i.e. with meta tensors.
            torch.library.impl_abstract(qualname, func=self.meta)

            # Register the concrete implementation.
            torch.library.impl(qualname, self.target, func=self.forward)