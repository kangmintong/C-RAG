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
import logging
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
import torch

class BAAIEval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)

        self.corpus = corpus
        self.cache_file_dir = 'outputs/baai/'

    def get_baai_embeddings(self, cur_corpus, task_name):
        out_path = os.path.join(self.cache_file_dir,f'{task_name}_baai_embedding.npy')
        if os.path.exists(out_path):
            logger.info(f'{out_path} exists, directly loading it')
            embeddings = np.load(out_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
            model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
            model.eval()

            embeddings = []
            with torch.no_grad():
                for entry in tqdm(cur_corpus):
                    doc = [entry['contents']]
                    encoded_input = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')
                    model_output = model(**encoded_input)
                    embedding = model_output[0][:, 0]
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding[0])
            embeddings = np.array(embeddings)
            np.save(out_path,embeddings)
            logger.info(f'save embedding to {out_path}')
        logger.info(f'embedding {out_path} dimension: {embeddings.shape}')
        return embeddings

    def compute_similarity(self, query_embedding, embeddings):
        similarity_scores = np.matmul(embeddings, np.transpose(query_embedding))
        return similarity_scores


    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:

        cur_corpus = self.corpus.filter(lambda x: x['task_name'] == task_names[0])
        embeddings = self.get_baai_embeddings(cur_corpus, task_names[0])

        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
        model.eval()

        scores_all = []

        logger.info(f'len(queries): {len(queries)}')
        # logger.info(f'query_embedding.shape: {query_embeddings.shape}')
        logger.info(f'embeddings.shape: {embeddings.shape}')

        start_time = time.time()
        query_embeddings = []
        with torch.no_grad():
            for query in tqdm(queries):
                encoded_input = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                model_output = model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0][0]
                query_embeddings.append(sentence_embeddings)
        end_time = time.time()

        logger.info(f'Wall clock time of query embeddings computation: {end_time-start_time}')

        scores_all = np.matmul(query_embeddings, embeddings.transpose())

        # for idx, query in tqdm(enumerate(queries)):
        #     query_embedding = query_embeddings[idx]
        #     scores = self.compute_similarity(query_embedding, embeddings)
        #     scores_all.append(scores)
        # scores_all = np.array(scores_all)

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