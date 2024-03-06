for method in llama-7b_BM25 llama-7b_BAAI llama-7b_OpenAI llama-7b_triever-base
do
  python conformal_risk_computation/conformal_distribution_shift_generation_risk.py --llm_retrievalmodel $method --datasets aeslc common_gen dart e2e_nlg --n_rag 15 --num_gen 1 --gen_thresh 0
done