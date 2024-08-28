#!/bin/bash

# Base command
base_cmd="python sparsedoping/greedyopt.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --histogram_path /data_persistent2/jamesliu/sparsedoping/Mistral-7B/histograms \
    --base_step_size 0.05 \
    --activations_path /data_persistent2/jamesliu/sparsedoping/Mistral-7B/activations/activations.pt" 


#     --activations_path /scratch/jamesliu/sparsedoping/Llama-2-7B/activations/activations.pt" 



# Number of layers and GPUs
# 7/8b: 32, 13b: 40, 70b: 80

num_layers=32
num_gpus=8

# Function to get number of running background jobs
get_job_count() {
    jobs -p | wc -l
}

# Loop through all layers
for layer in $(seq 0 $((num_layers-1))); do
    # Calculate which GPU to use
    gpu=$((layer % num_gpus))
    
    # Construct the full command
    cmd="CUDA_VISIBLE_DEVICES=$gpu $base_cmd \
        --layer_idx $layer \
        --output_path \"/data_persistent2/jamesliu/sparsedoping/Mistral-7B/lookup/layer-$layer/results.csv\""
    
    # Wait until we have less than num_gpus jobs running
    while [ $(get_job_count) -ge $num_gpus ]; do
        sleep 5
    done
    
    # Run the command in the background
    echo "Starting layer $layer on GPU $gpu"
    eval $cmd &
done

# Wait for all background processes to complete
wait

echo "All layer optimizations complete."
