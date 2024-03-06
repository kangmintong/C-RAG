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
from utils import save_json_to_file, get_method2name, get_method2method, get_num_points, empirical_risk2alpha_func

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute and plot the conformal generation risks of one retrieval model.')

    parser.add_argument('--llm_retrievalmodel', type=str, choices=['llama-7b_BM25', 'llama-7b_BAAI', 'llama-7b_OpenAI', 'llama-7b_triever-base'])
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--num_gen', type=int, default=1)
    parser.add_argument('--gen_thresh', type=int, default=0)
    parser.add_argument('--n_rag_list', nargs='+')

    args = parser.parse_args()

    method2name = get_method2name()
    method2method = get_method2method()

    datasets = args.datasets
    method = args.llm_retrievalmodel

    num_point_dict, sample_size_dict = get_num_points()


    for dataset in datasets:
        num_simulation_point = num_point_dict[dataset]
        sample_size_per_point = sample_size_dict[dataset]

        num_gen = args.num_gen
        gen_thresh = args.gen_thresh
        n_rag_list = args.n_rag_list

        alphas = {}
        min_risk, max_risk = 10e5, -10e5
        alpha_list = []

        for n_rag in n_rag_list:
            path = f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{num_gen}_{gen_thresh}_calibration_result.json'
            results = json.load(open(path, 'r', encoding='utf-8'))
            results = list(results.values())[0][0]
            results = np.array(results)
            alpha_list.append(empirical_risk2alpha_func(1.-np.mean(results), N_cal=len(results), delta=0.1))
            if alpha_list[-1]<min_risk:
                min_risk = alpha_list[-1]
            if alpha_list[-1]>max_risk:
                max_risk = alpha_list[-1]
        alphas[method] = alpha_list

        data = {
            'N_rag': n_rag_list,
             method: alphas[method],
        }

        df = pd.DataFrame(data)

        # Set the style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10,8))

        color_dict = {method: 'black'}

        for metric, color in color_dict.items():
            plt.plot(df['N_rag'], df[metric], marker='^', color=color, label=r'Certified Conformal Risk $\alpha_{\text{rag}}$', markersize=20)


        simulation_points = []
        for n_rag in n_rag_list:
            results = json.load(open(f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{num_gen}_{gen_thresh}_calibration_result.json', 'r', encoding='utf-8'))
            results = list(results.values())[0][0]
            results = np.array(results)

            simulation_points_temp = []
            for idx_point in range(num_simulation_point):
                test_list = random.sample(list(range(len(results))),sample_size_per_point)
                new_res = 1. - np.mean(results[test_list])
                simulation_points_temp.append(new_res)
            simulation_points_temp = np.array(simulation_points_temp)
            for x in simulation_points_temp:
                if x < min_risk:
                    min_risk = x
            plt.scatter(x=[n_rag]*len(simulation_points_temp),y=simulation_points_temp,color='gray',alpha=0.7,s=100)

        fontsize=45

        plt.ylim([min_risk-0.001, max_risk+0.02])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter('{x:3<0.2f}')

        plt.tight_layout()
        print('save figure at {}'.format({f'./figs/{dataset}_{method2name[method]}_conformal_generation_risk.jpg'}))
        plt.savefig(f'./figs/{dataset}_{method2name[method]}_conformal_generation risk.jpg',dpi=800)