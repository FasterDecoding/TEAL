import sys,os
# sys.path.append('../')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'utils'))

import torch
from tqdm import tqdm
import os
import argparse



if __name__ == "__main__":
    from utils.utils import get_tokenizer, get_sparse_model
    from utils.eval_ppl import eval_ppl

    from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
    from teal.model import MistralSparseForCausalLM, MistralSparseConfig

    from utils.data import get_dataset

    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)

    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",help='Name of the model to use')
    parser.add_argument('--teal_path', type=str, required=True,help='Path to the teal input')
    parser.add_argument('--greedy_flag', action='store_true', help='Flag for greedy')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity level')
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(args.model_name, device="auto", histogram_path=os.path.join(args.teal_path, "histograms"))

    dataset = get_dataset(
        "tatsu-lab/alpaca",
        subset=None,
        split="train",
        size=250
    )


    print("Evaluating dense PPL")
    print("="*40)
    dense_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"PPL: {dense_ppl}")


    print("Evaluating sparse PPL at sparsity level: ", args.sparsity)
    print("="*40)
    if args.greedy_flag:
        print("Evaluating greedy PPL")
        greedy_path = os.path.join(args.teal_path, "lookup")
        model.load_greedy_sparsities(greedy_path, args.sparsity)
    else:
        print("Evaluating uniform PPL")
        model.set_uniform_sparsity(args.sparsity)

    sparse_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"PPL: {sparse_ppl}")

    print("="*40)