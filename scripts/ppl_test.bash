# TEAL_PATH is the output path specified in grab_acts.py

CUDA_VISIBLE_DEVICES=0 python teal/ppl_test.py --model_name meta-llama/Llama-2-7b-hf --teal_path $OUTPUT_PATH --sparsity 0.5


