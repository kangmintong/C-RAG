import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from scipy.stats import binom
from typing import List, Union, Optional, Tuple, Mapping, Dict
import os
import json
from scipy.optimize import fsolve

def save_json_to_file(objects: Union[List, dict], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list), 'Only list can be saved in line by line format'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')

def get_method2name():
    method2name = {}
    method2name['llama-7b_BM25'] = 'BM25'
    method2name['llama-7b_triever-base'] = 'LLM-R'
    method2name['llama-7b_OpenAI'] = 'openai'
    method2name['llama-7b_BAAI'] = 'baai'
    return method2name


def get_method2method():
    method2method = {}
    method2method['llama-7b_triever-base'] = 'llm-retriever-base'
    method2method['llama-7b_BM25'] = 'bm25'
    method2method['llama-7b_OpenAI'] = 'openai'
    method2method['llama-7b_BAAI'] = 'baai'
    return method2method

def get_num_points():
    num_point_dict = {'aeslc': 100, 'common_gen': 100, 'dart': 100, 'e2e_nlg': 100}
    sample_size_dict = {'aeslc': 400, 'common_gen': 400, 'dart': 400, 'e2e_nlg': 400}
    return num_point_dict, sample_size_dict

def get_color_dict(retrieval_methods):
    colors = {'llama-7b_BM25': 'royalblue', 'llama-7b_triever-base': 'lightcoral',  'llama-7b_OpenAI': 'darkviolet', 'llama-7b_BAAI': 'olivedrab'}
    color_dict = {}
    for method in retrieval_methods:
        color_dict[method] = colors[method]
    return color_dict

def get_max_x_all():
    max_x_all = {'aeslc': 11, 'common_gen': 21, 'dart': 21, 'e2e_nlg': 21}
    return max_x_all

def get_num_points_distribution_shift():
    num_point_dict = {'aeslc': 100, 'common_gen': 100, 'dart': 100, 'e2e_nlg': 100}
    sample_size_dict = {'aeslc': 30, 'common_gen': 30, 'dart': 30, 'e2e_nlg': 30}
    simulate_num = 30000
    return num_point_dict, sample_size_dict, simulate_num

def h1(a,b):
    return a * np.log(a/b) + (1-a) * np.log((1-a)/(1-b))

def solve_inverse_1_func(r_hat, N_cal, delta):
    def func(a):
        ret = np.exp(-N_cal * h1(min(a,r_hat), a)) - delta
        return ret
    root = fsolve(func, [r_hat+0.02])[0]
    return root

def solve_inverse_2_func(r_hat, N_cal, delta):
    def func(a):
        ret = binom.cdf(np.ceil(N_cal * r_hat), N_cal, a) - delta / np.exp(1)
        return ret
    root = fsolve(func, [r_hat+0.02])[0]
    return root

def empirical_risk2alpha_func(r_hat, N_cal, delta):
    delta = delta
    alpha_term_1 = solve_inverse_1_func(r_hat, N_cal, delta)
    alpha_term_2 = solve_inverse_2_func(r_hat, N_cal, delta)
    alpha = min(alpha_term_1, alpha_term_2)
    if alpha>1.0:
        alpha=1.0
    return alpha

def cal_dist_shift_bound(r_hat, dist, N, var, delta):
    r_hat_overline = r_hat + dist**2 * (2-dist**2) * (1-r_hat) + 2 * dist * (1-dist**2) * np.sqrt(2-dist**2) * np.sqrt(var)
    # print(f'r_hat_overline 1: {r_hat_overline}')

    # finite-sample error
    r_hat_overline += (1-dist**2) * ((1-dist**2) / np.sqrt(2*N) + 2 * dist * np.sqrt(2 * (2 - dist**2)) / np.sqrt(N-1)) * np.sqrt(np.log(4/delta)) + np.sqrt(np.log(8/delta) / 2 / N)

    # print(f'r_hat_overline 2: {r_hat_overline}')
    # risk_shift = empirical_risk2alpha_newton(r_hat_overline, N, delta)
    risk_shift = empirical_risk2alpha_func(r_hat_overline, N, delta)
    return risk_shift

def compute_hellinger_distance(vec1, vec2):
    dist = 1.
    for idx in range(len(vec1)):
        dist -= np.sqrt(vec1[idx]) * np.sqrt(vec2[idx])
    dist = np.sqrt(dist)
    return dist

def compute_p_value(alpha_desired, r_hat, N_cal):
    term1 = np.exp(-N_cal * h1(min(alpha_desired,r_hat), alpha_desired))
    term2 = binom.cdf(np.ceil(N_cal * r_hat), N_cal, alpha_desired) * np.exp(1)
    p = min(term1, term2)
    if r_hat > alpha_desired:
        return 1.0
    return p

def FWER(p_values, delta):
    keys = p_values.keys()
    n_tol = len(keys)
    valid_or_not = {}
    # print(p_values)
    for key in p_values.keys():
        if p_values[key] < delta / n_tol:
            valid_or_not[key] = True
        else:
            valid_or_not[key] = False
    return valid_or_not

def compute_conformal_risk(results):
    return empirical_risk2alpha_func(np.mean(results), len(results), delta=0.1)