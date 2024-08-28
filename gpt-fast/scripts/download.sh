
model=meta-llama/Llama-2-7b-chat-hf

path=/scratch/jamesliu/checkpoints

python scripts/download.py --repo_id $model --path $path
