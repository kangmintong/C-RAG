# OUTPUT_DIR: results stored at the dir
# N_rag: number of retrieved examples
# llm_num_gen: number of generations
# llm_gen_threshold: accept similarity threshold during generations (Note: it is scaled by 100 and represented by an integer)

for n in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
for ngen in 1
  do
    GPUs=0,1,2,3 OUTPUT_DIR=outputs/bm25 N_rag=$n llm_num_gen=$ngen llm_gen_threshold=0 bash run/conformal_parallel.sh  BM25
  done
done