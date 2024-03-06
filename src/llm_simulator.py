import os
import json
import numpy as np

from typing import Dict, List, Optional
from transformers import AutoTokenizer
from datasets import Dataset

from config import Arguments
from logger_config import logger
from tasks import parse_decoded_text_by_task, get_metric_name_by_task_name, get_possible_answers_by_task_name
from evaluation.metrics import compute_metrics
from llms import BaseLLM
from model_utils import parse_model_id
from data_utils import save_llm_decoding_results
from utils import save_json_to_file, DictTrie, build_trie, wait_until_all_files_show_up, get_path_simulation_output

from rouge import Rouge
from tqdm import tqdm
from evaluation.metrics import simple_accuracy, acc_and_f1, f1_score, squad, trivia_qa
from evaluation import qa_utils
import random

class LLMSimulator:
    def __init__(self, args: Arguments, llm: BaseLLM):
        self.args = args
        self.llm = llm
        self.model_id: str = parse_model_id(self.args.model_name_or_path)
        self.llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)

    def calibrate_single_task(self, eval_dataset: Dataset, task_name: str):
        out_path: str = get_path_simulation_output(self.args, self.llm_model_id, self.model_id, task_name)
        # if os.path.exists(out_path):
        #     logger.info('Task {} has already been evaluated'.format(task_name))
        #     return

        task_ds_ori: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)

        # split task_ds into calibration and test set
        # logger.info(f'number of samples (calibration and test): {len(task_ds)}')
        splitted_task_ds = task_ds_ori.train_test_split(test_size=self.args.cali_ratio, seed=0, shuffle=True)
        task_ds_ori, _ = splitted_task_ds['train'], splitted_task_ds['test']

        num_total = len(task_ds_ori)
        simulations_results = []

        for idx_points in range(self.args.num_simulation_points):
            seed = random.randint(0, 1000000)
            splitted_task_ds = task_ds_ori.train_test_split(test_size=self.args.sample_size_per_point/num_total, seed=seed, shuffle=True)
            _, task_ds = splitted_task_ds['train'], splitted_task_ds['test']
            # logger.info(f'number of samples for calibration: {len(task_ds)}')

            # logger.info('Task: {}, # of examples: {}'.format(task_name, len(task_ds)))
            if len(task_ds) == 0:
                logger.error('No examples for task: {}'.format(task_name))
                return

            sharded_task_ds = task_ds.shard(num_shards=self.args.world_size, index=self.args.process_index, contiguous=True)
            logger.info('Worker {} needs to process {} examples'.format(self.args.process_index, len(sharded_task_ds)))

            queries: List[str] = sharded_task_ds['query']
            input_prompts: List[str] = sharded_task_ds['input_prompt']
            options_list: List[List[str]] = sharded_task_ds['options']
            answers: List = sharded_task_ds['answers']
            assert len(input_prompts) == len(queries)
            assert all(not q.endswith('\n') for q in queries)
            assert all(len(options) == len(options_list[0]) for options in options_list)

            # prompt may be empty in the zero-shot setting

            if self.args.llm_k_shot>0:
                input_texts: List[str] = [
                    '{}\n\n{}\n'.format(prompt, query) if prompt else '{}\n'.format(query) for prompt, query in
                    zip(input_prompts, queries)
                ]
                first = 1
                for prompt, query in zip(input_prompts, queries):
                    if first:
                        # print(prompt)
                        first = 0
            else:
                input_texts: List[str] = [
                    '{}\n'.format(query) for prompt, query in
                    zip(input_prompts, queries)
                ]
            possible_answers: Optional[List[str]] = get_possible_answers_by_task_name(task_name)

            # Compute empirical risks for a given configuration (args.N_rag,args.llm_num_gen,args.llm_gen_threshold)
            generations_list, raw_scores = self.sampling_procedure([self.args.llm_k_shot,self.args.llm_num_gen,self.args.llm_gen_threshold], task_name, input_texts, options_list, answers, possible_answers)
            save_json_to_file({
                'raw_scores': raw_scores,
            }, self._get_tmp_path(self.args.process_index, task_name))
            if self.args.process_index <= 0:
                wait_until_all_files_show_up(
                    [self._get_tmp_path(worker_idx, task_name) for worker_idx in range(self.args.world_size)]
                )
                IsFirst = 1
                for worker_idx in range(self.args.world_size):
                    tmp_path: str = self._get_tmp_path(worker_idx, task_name)
                    tmp_results: Dict = json.load(open(tmp_path, 'r', encoding='utf-8'))
                    if IsFirst == 1:
                        raw_scores_all = np.array(tmp_results['raw_scores'])
                        IsFirst = 0
                    else:
                        raw_scores_all = np.concatenate((raw_scores_all, np.array(tmp_results['raw_scores'])), axis=1)

                simulations_results.append(np.mean(raw_scores_all))
                # raw_scores_all = raw_scores_all.tolist()
                # calibration_results[str((self.args.llm_k_shot, self.args.llm_num_gen, self.args.llm_gen_threshold))] = raw_scores_all

        simu_res_json = {}
        simu_res_json[str((self.args.llm_k_shot, self.args.llm_num_gen, self.args.llm_gen_threshold))] = simulations_results
        save_json_to_file(simu_res_json, out_path)

    def sampling_procedure(self, lambda_config, task_name, input_texts, options_list, answers, possible_answers):
        # lambda_config: args.N_rag,args.llm_num_gen,args.llm_gen_threshold
        generation_list = []
        raw_scores = []
        for _ in range(lambda_config[1]):
            if len(options_list[0]) <= 1:
                # classification or open-ended generation tasks
                prefix_trie: Optional[DictTrie] = None
                if possible_answers and self.args.llm_constrained_decoding:
                    tokenizer = AutoTokenizer.from_pretrained(self.args.llm_model_name_or_path)
                    possible_answers = ['{}\n'.format(ans) for ans in possible_answers]
                    prefix_trie: DictTrie = build_trie(tokenizer=tokenizer, output_texts=possible_answers)
                    logger.info('Task: {}, constrained generation targets: {}'.format(task_name, possible_answers))
                decoded_texts: List[str] = self.llm.batch_decode(input_texts, prefix_trie=prefix_trie)
            else:
                # multiple-choice tasks
                assert len(options_list[0]) == len(possible_answers)
                choices: List[str] = sum(options_list, [])
                scoring_inputs = sum(
                    [[input_text.strip() for _ in range(len(possible_answers))] for input_text in input_texts], [])
                scores: List[float] = self.llm.batch_score(scoring_inputs, choices, delimiter='\n')
                answer_indices = np.argmax(np.array(scores).reshape(-1, len(possible_answers)), axis=1)
                decoded_texts: List[str] = [possible_answers[idx] for idx in answer_indices]
            parsed_decoded_texts: List[str] = [
                parse_decoded_text_by_task(decoded_text, task_name) for decoded_text in decoded_texts
            ]
            generation_list.append(parsed_decoded_texts)

            if max(len(answer) for answer in answers) == 1:
                # single answer
                answers: List[str] = [answer[0] for answer in answers]

            metric_name: str = get_metric_name_by_task_name(task_name)
            scores_cur: List = self.compute_scores_for_instances(metric=metric_name, labels=answers, preds=parsed_decoded_texts)
            raw_scores.append(scores_cur)

        return generation_list, raw_scores

    def compute_scores_for_instances(self, metric, labels, preds):
        assert len(preds) == len(labels)

        if metric == 'rouge':
            rls = []
            r = Rouge()
            for i in range(len(labels)):
                if '\n' not in preds[i]: preds[i] += '\n'
                if '\n' not in labels[i]: labels[i] += '\n'  # avoid empty string
                scores = r.get_scores(preds[i], labels[i])[0]
                rls.append(scores["rouge-l"]['f'])
            return rls
        elif metric == 'simple_accuracy':
            if isinstance(preds[0], str):
                labels = [label.lower().strip() for label in labels]
                preds = [pred.lower().strip() for pred in preds]
            res = [int(preds[i] == labels[i]) for i in range(len(preds))]
            return res
        elif metric == 'acc_and_f1':
            if isinstance(preds[0], str):
                labels = [label.lower().strip() for label in labels]
                preds = [pred.lower().strip() for pred in preds]
            res = [int(preds[i] == labels[i]) for i in range(len(preds))]
            return res
        elif metric == 'f1':
            if isinstance(preds[0], str):
                labels = [label.lower().strip() for label in labels]
                preds = [pred.lower().strip() for pred in preds]
            res = [int(preds[i] == labels[i]) for i in range(len(preds))]
            return res
        elif metric == 'squad':
            labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
            preds = [qa_utils.normalize_squad(p) for p in preds]
            em = [
                qa_utils._metric_max_over_ground_truths(qa_utils._exact_match_score, t, p)
                for p, t in zip(preds, labels)
            ]
            return em
        elif metric == 'trivia_qa':
            labels = [[qa_utils.normalize_trivia_qa(t) for t in u] for u in labels]
            preds = [qa_utils.normalize_trivia_qa(p) for p in preds]
            em = [
                qa_utils._metric_max_over_ground_truths(qa_utils._exact_match_score, t, p)
                for p, t in zip(preds, labels)
            ]
            return em
        else:
            raise ValueError(metric)


    def _get_tmp_path(self, worker_idx: int, task_name: str) -> str:
        tmp_dir = self.args.output_dir if self.args.world_size <= 1 else 'tmp/'
        llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)
        return '{}/{}/{}_{}.json'.format(tmp_dir, llm_model_id, task_name, worker_idx)