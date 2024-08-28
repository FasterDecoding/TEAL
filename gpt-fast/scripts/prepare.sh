
model=meta-llama/Llama-2-7b-hf

path=/scratch/jamesliu/checkpoints

python scripts/download.py --repo_id $model --path $path && python scripts/convert_hf_checkpoint.py --checkpoint_dir $path/$model
