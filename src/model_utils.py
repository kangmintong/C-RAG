import os

from datasets import Dataset

from llms import BaseLLM, GPT2, GPTNeo, Llama
from evaluation import BaseEval, RandomEval, DenseEval, OpenaiEval, BM25Eval, BAAIEval
from config import Arguments
from logger_config import logger


def build_llm(args: Arguments) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'gpt2' in model_name_or_path:
        if args.llm_max_input_length >= 1024:
            args.llm_max_input_length -= max(args.llm_max_decode_length, 128)
            logger.warning('GPT2 models cannot handle sequences longer than 1024. '
                           'set to {}'.format(args.llm_max_input_length))
        llm = GPT2(args=args, model_name_or_path=model_name_or_path)
    elif 'gpt-neo' in model_name_or_path:
        llm = GPTNeo(args=args, model_name_or_path=model_name_or_path)
    elif 'llama' in model_name_or_path:
        llm = Llama(args=args, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))

    return llm


def build_eval_model(args: Arguments, corpus: Dataset) -> BaseEval:
    model_name_or_path: str = args.model_name_or_path
    if model_name_or_path == 'random':
        return RandomEval(args=args, corpus=corpus)
    elif model_name_or_path == 'BM25':
        return BM25Eval(args=args, corpus=corpus)
    elif model_name_or_path == 'OpenAI':
        return OpenaiEval(args=args, corpus=corpus)
    elif model_name_or_path == 'BAAI':
        return BAAIEval(args=args, corpus=corpus)
    elif 'llm-retriever-base' in model_name_or_path:
        # LLM-R
        return DenseEval(args=args, corpus=corpus)
    else:
        raise TypeError(f'retrieval method {model_name_or_path} not implemented')


def parse_model_id(model_name_or_path: str) -> str:
    if model_name_or_path in ['random', 'BM25', 'Openai']:
        return model_name_or_path
    return os.path.basename(model_name_or_path.strip('/'))[-12:]
