# should get data, tokenizer, then layer by layer grab activations and save histograms + the activations themselves



import argparse
import os

import torch
import transformers

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))


from utils.utils import get_tokenizer, get_sparse_model

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
from teal.model import MistralSparseForCausalLM, MistralSparseConfig

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)

AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",help='Name of the model to use')
parser.add_argument('--output_path', type=str, required=True,help='Path to the output') # contains 1. model itself, 2. histograms, 3. activations
args = parser.parse_args()

tokenizer = get_tokenizer(args.model_name)
model = get_sparse_model(args.model_name, device="auto", histogram_path=os.path.join(args.output_path, "histograms"), grab_acts=True)

from utils.data import get_dataset
from tqdm import tqdm
import gc

# build histograms
dataset = get_dataset(
    "tatsu-lab/alpaca",
    subset=None,
    split="train",
    size=300
)
text = ""
for sample in tqdm(dataset):
    text += sample["text"] + "\n\n"

print(len(text))
bsz, seq_len = 10, 2048

encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")

input_ids = encodings.input_ids[:bsz,:].to(device="cuda:0")
print(input_ids.shape)

hidden_states = model.model.embed_tokens(input_ids)

attention_mask = None
position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
past_key_value=None
output_attentions = False
use_cache = False
cache_position=None
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)


act_path = os.path.join(args.output_path, "activations")
os.makedirs(act_path, exist_ok=True)

for i in tqdm(range(len(model.model.layers))):
    # for greedyopt
    torch.save(hidden_states, os.path.join(act_path, f"act_{i}.pt"))


    layer = model.model.layers[i]
    hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.data.device) 
    hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]

    layer.mlp.activation_module.find_histogram()
    layer.self_attn.activation_module.find_histogram()
    layer.mlp.activation_module.save_histogram()
    layer.self_attn.activation_module.save_histogram()

    del layer.mlp.activation_module.activations
    del layer.self_attn.activation_module.activations
    
    model.model.layers[i] = None

    gc.collect()
    torch.cuda.empty_cache()