for method in llama-7b_BM25 llama-7b_BAAI llama-7b_OpenAI llama-7b_triever-base
do
  python conformal_risk_computation/conformal_generation_risk_similarity_threshold.py \
   --llm_retrievalmodel $method --datasets aeslc common_gen dart e2e_nlg --n_rag 5 --num_gen 20 --thresh_list 10 9 8 7 6
done