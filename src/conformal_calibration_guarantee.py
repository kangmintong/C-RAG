from datasets import Dataset, load_dataset, DownloadMode
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from llms import BaseLLM
from model_utils import build_llm
from data_utils import log_task_statistics
from llm_calibrator import LLMCalibrator
from inference.inference_utils import get_prompt_save_path
from model_utils import parse_model_id
import json
import numpy as np
from utils import get_path_calibration_output, get_path_conformal_guarantee_res, save_json_to_file

from scipy.stats import binom
from tqdm import tqdm

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

def h1(a,b):
    return a * np.log(a/b) + (1-a) * np.log((1-a)/(1-b))
def empirical_risk2alpha(risk, delta):
    r_hat = risk.mean()
    N_cal = len(risk)
    thresh_1, thresh_2, num = 1, 1, int(1e6)
    for num_ in tqdm(range(1,num)):
        alpha_ = 1.0 * num_ / num
        cur_1 = np.exp(-N_cal * h1(alpha_, r_hat))
        if abs(cur_1 - delta) < thresh_1:
            thresh_1 = abs(cur_1 - delta)
            alpha_term_1 = alpha_
        cur_2 = binom.cdf(np.ceil(N_cal * r_hat), N_cal, alpha_)
        if abs(cur_2 - delta / np.exp(1)) < thresh_2:
            thresh_2 = abs(cur_2 - delta / np.exp(1))
            alpha_term_2 = alpha_

    print(f'empirical risk: {r_hat}')
    print(f'alpha_term_1: {alpha_term_1}')
    print(f'alpha_term_2: {alpha_term_2}')
    alpha = max(alpha_term_1, alpha_term_2)
    return alpha
def compute_risk(raw_scores, delta, args):
    # raw scores: rouge-l or accuracy
    raw_scores = np.array(raw_scores)
    print(f'raw_scores.shape before aggregation: {raw_scores.shape}')
    if args.aggregation_criteria=='max':
        raw_scores = raw_scores.max(axis=0)
    elif args.aggregation_criteria=='avg':
        raw_scores = raw_scores.mean(axis=0)
    else:
        raise TypeError(f"the aggregation criteria {args.aggregation_criteria} is not implemented")
    print(f'raw_scores.shape after aggregation (quality scores of generations in the set): {raw_scores.shape}')
    empirical_risks = 1. - raw_scores
    conformal_alpha = empirical_risk2alpha(empirical_risks,delta)
    return conformal_alpha

def main():
    # columns: query_id / query / answers / task_name / input_prompt
    eval_dataset: Dataset = load_dataset(
        'json', data_files=get_prompt_save_path(args), split='train',
        download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
        args.llm_eval_tasks = sorted(eval_dataset.unique('task_name'))
        logger.info('Eval all {} tasks'.format(len(args.llm_eval_tasks)))

    delta = args.delta


    for task_name in args.llm_eval_tasks:
        config2risk = {}
        model_id: str = parse_model_id(args.model_name_or_path)
        llm_model_id: str = parse_model_id(args.llm_model_name_or_path)
        conformal_res_path: str= get_path_calibration_output(args, llm_model_id, model_id, task_name)
        raw_scores_all = json.load(open(conformal_res_path, 'r', encoding='utf-8'))
        configs_keys = raw_scores_all.keys()

        for config in configs_keys:
            raw_scores = raw_scores_all[config]
            N = len(raw_scores[0])
            config2risk[config] = compute_risk(raw_scores, delta, args)
            logger.info(f'task: {task_name}, sample size: {N}')
            logger.info(f'config2risk: {config2risk}')

        out_path = get_path_conformal_guarantee_res(args, llm_model_id, model_id, task_name)
        save_json_to_file(config2risk, out_path)
    logger.info('Done')


if __name__ == '__main__':
    main()
