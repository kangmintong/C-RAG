import json
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from scipy.stats import binom
from typing import List, Union, Optional, Tuple, Mapping, Dict
import os
from numpy import linalg
import numpy as np
import argparse
from utils import save_json_to_file, get_method2name, get_method2method, get_num_points, empirical_risk2alpha_func, compute_conformal_risk

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute and plot the conformal generation risks of one retrieval model.')

    parser.add_argument('--llm_retrievalmodel', type=str, choices=['llama-7b_BM25', 'llama-7b_BAAI', 'llama-7b_OpenAI', 'llama-7b_triever-base'])
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--n_rag', type=int, default=5)
    parser.add_argument('--gen_thresh', type=int, default=0)
    parser.add_argument('--num_gen_list', nargs='+')

    args = parser.parse_args()

    method2name = get_method2name()
    method2method = get_method2method()

    datasets = args.datasets
    method = args.llm_retrievalmodel

    num_point_dict, sample_size_dict = get_num_points()

    num_list = []
    for idx in range(len(args.num_gen_list)):
        num_list.append(int(args.num_gen_list[idx]))
    max_num_gen = max(num_list)

    random.seed(1)

    for dataset in datasets:
        num_simulation_point = num_point_dict[dataset]
        sample_size_per_point = sample_size_dict[dataset]

        num_gen_list = args.num_gen_list
        num_gen_list = [int(item) for item in num_gen_list]
        gen_thresh = args.gen_thresh
        n_rag = args.n_rag

        alphas = {}
        min_risk, max_risk = 10e5, -10e5
        rand_list = []
        alpha_list = []
        for num_gen in num_gen_list:
            num_gen = int(num_gen)
            path = f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{max_num_gen}_{gen_thresh}_calibration_result.json'
            result = json.load(open(path, 'r', encoding='utf-8'))
            result = list(result.values())[0]
            result = np.array(result)
            rand_indx = random.sample(tuple(list(range(0,max_num_gen))), num_gen)
            result = result[rand_indx,:]
            rand_list.append(rand_indx)
            result = result.max(axis=0)
            results = 1. - result

            alpha_list.append(compute_conformal_risk(results))
            if alpha_list[-1] < min_risk:
                min_risk = alpha_list[-1]
            if alpha_list[-1] > max_risk:
                max_risk = alpha_list[-1]
        alphas[method] = alpha_list

        data = {
            'num_gen': num_gen_list,
            method: alphas[method],
        }

        df = pd.DataFrame(data)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10, 8))

        color_dict = {method: 'black'}

        for metric, color in color_dict.items():
            plt.plot(df['num_gen'], df[metric], marker='^', color=color, label=r'Certified Conformal Risk $\alpha_{\text{rag}}$', markersize=20)

        simulation_points = []

        for num_gen in num_gen_list:
            num_gen = int(num_gen)

            path = f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{max_num_gen}_{gen_thresh}_calibration_result.json'

            result = json.load(open(path, 'r', encoding='utf-8'))
            result = list(result.values())[0]
            result = np.array(result)
            result = result[rand_list[num_gen-1], :]
            result = result.max(axis=0)
            results = 1. - result

            simulation_points_temp = []
            for idx_point in range(num_simulation_point):
                test_list = random.sample(list(range(len(results))), sample_size_per_point)
                res = np.mean(results[test_list])
                simulation_points_temp.append(res)
            simulation_points_temp = np.array(simulation_points_temp)
            for x in simulation_points_temp:
                if x < min_risk:
                    min_risk = x

            plt.scatter(x=[num_gen] * len(simulation_points_temp), y=simulation_points_temp, color='gray', alpha=0.7, s=100)

        fontsize = 45

        plt.ylim([min_risk - 0.02, max_risk + 0.02])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax = plt.gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter('{x:3<0.2f}')

        plt.tight_layout()
        print('save figure at {}'.format({f'./figs/{dataset}_{method2name[method]}_multi_gen_conformal_bound_simulation.jpg'}))
        plt.savefig(f'./figs/{dataset}_{method2name[method]}_multi_gen_conformal_bound_simulation.jpg', dpi=800)