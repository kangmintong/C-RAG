# C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models [ICML 2024]

We provide the implementation of [C-RAG](https://arxiv.org/abs/2402.03181) in this repositary. 

C-RAG is the first framework to certify generation risks for RAG models. Specifically, C-RAG provides conformal risk analysis for RAG models and certify an upper confidence bound of generation risks, which is refered to as conformal generation risk. 
C-RAG also provides theoretical guarantees on conformal generation risks for general bounded risk functions under test distribution shifts. 
C-RAG proves that RAG achieves a lower conformal generation risk than that of a single LLM when the quality of the retrieval model and transformer is non-trivial. 
The intensive empirical results demonstrate the soundness and tightness of the conformal generation risk guarantees across four widely-used NLP datasets on four state-of-the-art retrieval models.

## Environment

Install PyTorch with correponding environment and CUDA version at [Pytorch Installation](https://pytorch.org/get-started/locally/).

Run ``pip install -r requirement.txt`` for installation of other neccessary packages in the repo.

## Pretrained models
For the supervised-finetuned biencoder-based retrieval model, we follow the implementation in [LLM-R](https://arxiv.org/abs/2307.07164) and provide the model checkpoint at [trained_retrieval_models](https://drive.google.com/file/d/1xOeCz3vt2piHuY00a5q4YCNhDkyCs0VF/view?usp=sharing).

Or you can download it by command:
```
gdown https://drive.google.com/uc?id=1xOeCz3vt2piHuY00a5q4YCNhDkyCs0VF
```

Then, put the folder ``trained_retrieval_models/`` under ``C-RAG/``.

## Dataset preparation
We evaluate C-RAG on four widely used NLP datasets, including AESLC, CommonGen, DART, and E2E. We preprocess the data and provide it at [data](https://drive.google.com/file/d/1JJC192wdOmXYZy_hXcGVrXOtMK2LWsv7/view?usp=sharing).

Or you can download it by command:
```
gdown https://drive.google.com/uc?id=1JJC192wdOmXYZy_hXcGVrXOtMK2LWsv7
```

Then, put the folder ``data/`` under ``C-RAG/``.

## Evaluate conformal generation risks in C-RAG

To compute the conformal generation risk, we need to (1) evaluate the raw risk scores for calibration instances following our constrained generation protocol, and (2) compute the conformal generation risks based on empirical risk statistics.

### (1) Evaluate raw risk scores for calibration instances

#### We compact the process in four scripts for four retrieval models

Evaluate raw risk scores for BM25 retrieval model:
```
sh scripts_raw_risk_scores/bm25.sh
```

Evaluate raw risk scores for BAAI/bge retrieval model:
```
sh scripts_raw_risk_scores/baai.sh
```

Evaluate raw risk scores for OpenAI/text-embedding-ada-002 retrieval model:
```
sh scripts_raw_risk_scores/openai.sh
```

Evaluate raw risk scores for LLM-R finetuned biencoder-based retrieval model:
```
sh scripts_raw_risk_scores/llm-r.sh
```

#### Exaplanation: we compact the following two steps in the scripts above:

1. Prepare the prompt via ``src/inference/generate_few_shot_prompt.py``: <br> Retrieve relevant examples and store the prompts at `` outputs/{METHOD}/{METHOD}_test_k{N_RAG}.jsonl.gz``
2. Evaluate the risks of prompts on calibration sets via ``src/conformal_calibration_empirical_risk.py``: <br> Evaluate the prompts and store results in ``outputs/{METHOD}/{LLM}_{METHOD}/``


### (2) Compute conformal generation risks

The conformal generation risk computation is based on empirical risk statistics stored at ``outputs/{METHOD}/{LLM}_{METHOD}/`` in step (1).

#### Conformal generation risk without distribution shifts
1. Compute conformal generation risks of a single retrieval model and compare it with the simulation results: 
```
sh scripts_conformal_generation_risk/run_conformal_generation_risk.sh
```
2. Compare conformal generation risks of different retrieval models (after running step 1 for corresponding methods): 
```
sh scripts_conformal_generation_risk/run_conformal_generation_risk_comparisons.sh
```

#### Conformal generation risk with distribution shifts
1. Compute conformal generation risks of a single retrieval model and compare it with the simulation results: 
```
sh scripts_conformal_generation_risk/run_conformal_distribution_shift_generation_risk.sh
```
2. Compare conformal generation risks of different retrieval models (after running step 1 for corresponding methods): 
```
sh scripts_conformal_generation_risk/run_conformal_distribution_shift_generation_risk_comparisons.sh
```

#### Conformal generation risk with multi-dimensional RAG configurations
```
sh scripts_conformal_generation_risk/run_conformal_generation_risk_multi_dim_config.sh
```

#### Valid configurations given desired risk levels
```
sh scripts_conformal_generation_risk/run_conformal_generation_risk_valid_config.sh
```

#### Additional evaluations with varying RAG configurations

Conformal generation risks with varying generation set sizes:
```
sh scripts_conformal_generation_risk/run_conformal_generation_risk_num_gen.sh
```

Conformal generation risks with varying generation similar thresholds:
```
sh scripts_conformal_generation_risk/run_conformal_generation_risk_similarity_threshold.sh
```


## Acknowledgement

The inference part in the repo is built on [LLM-R repo](https://github.com/microsoft/LMOps/tree/main/llm_retriever).

For any related questions or discussion, please contact ``mintong2@illinois.edu``.

If you find our paper and repo useful for your research, please consider cite:
```
@article{kang2024c,
  title={C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models},
  author={Kang, Mintong and G{\"u}rel, Nezihe Merve and Yu, Ning and Song, Dawn and Li, Bo},
  journal={arXiv preprint arXiv:2402.03181},
  year={2024}
}
```
