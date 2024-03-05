#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="random"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

LLM_MODEL_NAME_OR_PATH="huggyllama/llama-7b"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    LLM_MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/tasks/"
fi


EVAL_TASKS=('aeslc' 'common_gen' 'dart' 'e2e_nlg')

# generate data: input prompt (retrieved examples) and query for the specified task
PYTHONPATH=src/ python -u src/inference/generate_few_shot_prompt.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed 1234 \
    --fp16 \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split test \
    --llm_k_shot "${N_rag}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}"


nproc=$(echo -n "$GPUs" | wc -m)
nproc=`expr $nproc + 1`
nproc=`expr $nproc / 2`

# evaluate the empirical risks
CUDA_VISIBLE_DEVICES="${GPUs}" torchrun --nproc_per_node "${nproc}" --master_port=38765 src/conformal_calibration_empirical_risk.py \
    --llm_batch_size_per_device 4 \
    --llm_k_shot "${N_rag}" \
    --llm_num_gen "${llm_num_gen}" \
    --llm_gen_threshold "${llm_gen_threshold}" \
    --cali_ratio 0.5 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed 1234 \
    --fp16 \
    --do_llm_eval \
    --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split test \
    --llm_max_input_length 1024 \
    --llm_max_decode_length 64 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"