import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import types

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
)

from utils.utils import ActivationModule, Distribution, SparsifyFn, get_module_device

from transformers.modeling_flash_attention_utils import _flash_attention_forward

def _monkeypatch_self_attn(self_attn, file_path, grabbing_mode=False):
    self_attn.forward_old = self_attn.forward

    self_attn.forward = types.MethodType(_FA2_forward, self_attn)

    self_attn.file_path = file_path
    self_attn.grabbing_mode = grabbing_mode

    if not grabbing_mode:
        self_attn.distrs = {}
        self_attn.distrs['h1'] = Distribution(file_path, hidden_type='h1')
        self_attn.distrs['h2'] = Distribution(file_path, hidden_type='h2')

        self_attn.sparse_fns = nn.ModuleDict({
            'q': SparsifyFn(self_attn.distrs['h1']).to(get_module_device(self_attn)),
            'k': SparsifyFn(self_attn.distrs['h1']).to(get_module_device(self_attn)),
            'v': SparsifyFn(self_attn.distrs['h1']).to(get_module_device(self_attn)),
            'o': SparsifyFn(self_attn.distrs['h2']).to(get_module_device(self_attn))
        })

    self_attn.activation_module = ActivationModule(file_path)

    return self_attn


def _FA2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    past_key_value = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    activation_module = None,
    **kwargs,
): # -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # if isinstance(past_key_value, StaticCache):
    #     raise ValueError(
    #         "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
    #         "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
    #     )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # MONKEYPATCH HERE
    
    if self.grabbing_mode:
        self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    else: 
        x_q = self.sparse_fns['q'](hidden_states)

        x_k = self.sparse_fns['k'](hidden_states)
        x_v = self.sparse_fns['v'](hidden_states)

        query_states = self.q_proj(x_q)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)
        
    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        # logger.warning_once(
        #     f"The input hidden states seems to be silently casted in float32, this might be related to"
        #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #     f" {target_dtype}."
        # )
        print(f"Casting input hidden states to {target_dtype} (this should not be happening)")

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # NOTE: sliding window isn't tested for Mistral, please create an issue if something goes wrong
    # However, we don't ever utilize sequence lengths of more than 4096 for the methodology + evals
    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, position_ids=position_ids, dropout=dropout_rate, sliding_window=getattr(self, "sliding_window", None), is_causal=True
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:
        attn_output = self.sparse_fns['o'](attn_output)
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value