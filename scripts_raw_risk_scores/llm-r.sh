# OUTPUT_DIR: results stored at the dir
# N_rag: number of retrieved examples
# llm_num_gen: number of generations
# llm_gen_threshold: accept similarity threshold during generations (Note: it is scaled by 100 and represented by an integer)

# path_retrieval_model specifies path of locally trained retrieval model
# Huggingface transformer downloads or loads from cache from ~/.cache/huggingface/hub/models--llm_dir--llm_name (including blobs,refs,snapshots)

for n in 0 1 2 3 # 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
for ngen in 1
  do
    GPUs=0,1,2,3 OUTPUT_DIR=outputs/llm-retriever-base N_rag=$n llm_num_gen=$ngen llm_gen_threshold=0 bash run/conformal_parallel.sh  trained_retrieval_models/LLM-R/llm-retriever-base
  done
done