import os
from typing import List, Dict, Tuple
from datasets import Dataset

from evaluation.base_eval import BaseEval
from config import Arguments
from logger_config import logger
import tiktoken
from tqdm import tqdm
import numpy as np

import functools
import signal

import time
import numpy as np
from tqdm import tqdm
import os
import openai

import logging
from numpy.linalg import norm


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func

    return decorator


class AzureEmbedAPI(object):
    def __init__(self, engine_name='text-embedding-ada-002'):
        self.engine_name = engine_name

        self.cost = 0
        openai.api_key = "a5bee4a6d888497cbf4dff7b440cdb4b"
        openai.api_base = "https://lzn-openai.openai.azure.com/"
        self.model_name = "lzn-embedding-ada-002"

        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'
        logging.info("Pricing: 0.0004/1k tokens")  # 20 times cheaper than GPT-3.5
        self.prompt_price = 0.0004 / 1000

    def get_embeddings(self, messages):
        """
        Do text embedding by calling `openai.Embedding.create`
        Args:
            messages (`List[str]`): texts to be encode
        """

        total_tokens = 0

        response = []  # only contains embedding vectors

        for message in tqdm(messages):
            message = str(message)
            _response, _tokens = self.call(message)
            if _response is None:
                _response = [0] * 1536

            response.append(_response)
            total_tokens += _tokens

        self.cost += total_tokens * self.prompt_price

        return response

    @timeout(60)
    def _call(self, message):

        result = openai.Embedding.create(
            model=self.engine_name,
            engine=self.model_name,
            input=message,
        )

        embedding_result = result['data'][0]['embedding']
        num_tokens = result["usage"]['total_tokens']

        return embedding_result, num_tokens

    def call(self, message, retry=20):
        """
        A robust implementation for calling `openai.ChatCompletion.create`.
        Args:
            messages: messages conveyed to OpenAI.
            t: temperature. Set t=0 will make the outputs mostly deterministic.
            max_tokens: maximum tokens to generate for chat completion. Please look at https://platform.openai.com/docs/api-reference/chat/create for more information.

            retry: for sake of Error on OpenAI side, we try `retry + 1` times for a request if we do not get a response.
        """
        response = None
        num_tokens = 0
        for i in range(retry + 1):
            try:
                response, num_tokens = self._call(message)
                break
            except TimeoutError:
                logging.info(f"Seemingly openai is frozen, wait {i + 1}s and retry")
                time.sleep(i + 1)
            except Exception as e:
                logging.info("Error:", e)
                logging.info(type(e))
                # if isinstance(e, (openai.error.Timeout, openai.error.RateLimitError)):
                logging.info(f"wait {i + 1}s and retry")
                time.sleep(i + 1)

        if response is None:
            logging.info(f"try {retry + 1} but still no response, return None")
        return response, num_tokens

class OpenaiEval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)

        self.corpus = corpus
        self.cache_file_dir = 'outputs/openai/'

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_openai_embeddings(self, cur_corpus, task_name):
        out_path = os.path.join(self.cache_file_dir,f'{task_name}_openai_embedding.npy')
        if os.path.exists(out_path):
            logger.info(f'{out_path} exists, directly loading it')
            embeddings = np.load(out_path)
        else:
            embedding_module = AzureEmbedAPI()
            embeddings = []
            for entry in tqdm(cur_corpus):
                doc = entry['contents']
                embedding = embedding_module.get_embeddings([doc])
                embeddings.append(embedding[0])
            embeddings = np.array(embeddings)
            np.save(out_path,embeddings)
            logger.info(f'save embedding to {out_path}')
        logger.info(f'embedding {out_path} dimension: {embeddings.shape}')
        return embeddings

    @staticmethod
    def get_single_embedding(text):
        embedding_module = AzureEmbedAPI()
        embedding = embedding_module.get_embeddings([text])[0]
        return embedding
    def compute_similarity(self, query_embedding, embeddings):
        # similarity_scores = []
        # query_embedding = np.array(query_embedding)
        # for embedding in embeddings:
        #     embedding = np.array(embedding)
        #     similarity_scores.append(np.dot(query_embedding,embedding)/(norm(query_embedding)*norm(embedding)))
        similarity_scores = np.matmul(embeddings, np.transpose(query_embedding))
        return similarity_scores


    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:
        # cur_corpus = self.corpus
        num_tokens = 0
        cur_corpus = self.corpus.filter(lambda x: x['task_name'] == task_names[0])
        embeddings = self.get_openai_embeddings(cur_corpus, task_names[0])

        # normalize embeddings
        embeddings_norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        embeddings = embeddings / embeddings_norms

        embedding_module = AzureEmbedAPI()
        scores_all = []

        query_embeddings = embedding_module.get_embeddings(queries)

        for idx, query in tqdm(enumerate(queries)):
            query_embedding = query_embeddings[idx]
            scores = self.compute_similarity(query_embedding, embeddings)
            scores_all.append(scores)
        scores_all = np.array(scores_all)

        doc_ids = []
        for entry in cur_corpus:
            doc_ids.append(entry['id'])

        topk_index = scores_all.argsort(axis=1)[:, -k:]
        result = []
        for idx in range(len(topk_index)):
            cur_list = []
            for j in range(len(topk_index[idx])):
                doc_id_ = doc_ids[topk_index[idx][j]]
                doc_id_score = scores_all[idx][topk_index[idx][j]]
                cur_list.append((doc_id_score, doc_id_))
            result.append(cur_list)
        return result