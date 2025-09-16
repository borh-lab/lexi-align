#!/usr/bin/env bash

for model in "mistralai/Mistral-Nemo-Instruct-2407" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"; do
    time uv run python evaluations/xl-wa.py analyze --adapter outlines --model "$model" --temperature 0.0 -o "$(echo $model | sed 's|/|_|g')-EN-SL-v0.5.0-results" --num-train-examples 1 -vvv --seed 42 --lang-pairs EN-SL --model-dtype int8 --batch-size 4
done
