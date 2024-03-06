from typing import List, Dict, Tuple
from datasets import Dataset

from evaluation.base_eval import BaseEval
from config import Arguments
from logger_config import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os
import numpy as np

class BM25Eval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)

        self.corpus = corpus
        # initializing punctuations string
        self.punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        self.cache_file_dir = 'outputs/bm25/'

    def tokenize(self, str):
        str = str.lower()
        for ele in str:
            if ele in self.punc:
                str = str.replace(ele, "")
        doc_tokens = str.split(" ")
        return doc_tokens



    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:
        tokenized_corpus = []
        doc_ids = []
        logger.info('Start tokenizing corpus for BM25')
        cur_corpus = self.corpus.filter(lambda x: x['task_name'] == task_names[0])

        outpath_scores = os.path.join(self.cache_file_dir,f'{task_names[0]}_bm25_scores.npy')
        outpath_ids = os.path.join(self.cache_file_dir, f'{task_names[0]}_bm25_docids.npy')

        if os.path.exists(outpath_scores):
            logger.info(f'score file {outpath_scores} already exists, skip calculating')
            doc_scores_all = np.load(outpath_scores)
            doc_ids = np.load(outpath_ids)
        else:
            for entry in tqdm(cur_corpus):
                doc = entry['contents']
                doc_tokens = self.tokenize(doc)
                tokenized_corpus.append(doc_tokens)
                doc_ids.append(entry['id'])

            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores_all = []
            for query in tqdm(queries):
                tokenized_query = self.tokenize(query)
                doc_scores = bm25.get_scores(tokenized_query)
                doc_scores_all.append(doc_scores)

            doc_scores_all = np.array(doc_scores_all)
            doc_ids = np.array(doc_ids)

            np.save(outpath_scores, doc_scores_all)
            np.save(outpath_ids, doc_ids)

        topk_index = doc_scores_all.argsort(axis=1)[:,-k:]
        result = []
        for idx in range(len(topk_index)):
            cur_list = []
            for j in range(len(topk_index[idx])):
                doc_id_ = doc_ids[topk_index[idx][j]]
                doc_id_score = doc_scores_all[idx][topk_index[idx][j]]
                cur_list.append((doc_id_score, doc_id_))
            result.append(cur_list)
        return result
