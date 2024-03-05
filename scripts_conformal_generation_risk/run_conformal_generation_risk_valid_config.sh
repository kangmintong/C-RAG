for method in llama-7b_BM25 llama-7b_BAAI llama-7b_OpenAI llama-7b_triever-base
do
  python conformal_risk_computation/conformal_generation_risk_valid_config.py \
   --llm_retrievalmodel $method --datasets aeslc common_gen dart e2e_nlg --n_rag_list 0 1 2 3 4 5 6 7 8 9 --lambda_g_list 1 2 3 4 5 6 7 8 9 10
done