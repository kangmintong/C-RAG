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
    parser.add_argument('--gen_thresh', type=int, default=0)
    parser.add_argument('--n_rag_list', nargs='+')
    parser.add_argument('--lambda_g_list', nargs='+')

    args = parser.parse_args()

    method2name = get_method2name()
    method2method = get_method2method()

    datasets = args.datasets
    method = args.llm_retrievalmodel

    n_rag_list = np.array(args.n_rag_list)
    lambda_g_list = np.array(args.lambda_g_list)

    num_point_dict, sample_size_dict = get_num_points()

    for dataset in datasets:

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        xdata, ydata, zdata = [], [], []
        c = []
        random.seed(1)
        rand_list = []

        num_list = []
        for idx in range(len(lambda_g_list)):
            num_list.append(int(lambda_g_list[idx]))
        max_num_gen = max(num_list)


        for n_rag in n_rag_list:
            for num_gen in lambda_g_list:
                num_gen = int(num_gen)
                path = f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{max_num_gen}_{args.gen_thresh}_calibration_result.json'

                result = json.load(open(path, 'r', encoding='utf-8'))
                result = list(result.values())[0]
                result = np.array(result)
                rand_indx = random.sample(tuple(list(range(0, max_num_gen))), num_gen)

                result = result[rand_indx, :]
                rand_list.append(rand_indx)
                result = result.max(axis=0)
                results = 1. - result

                alpha_rag = empirical_risk2alpha_func(np.mean(results), N_cal=len(results), delta=0.1)

                xdata.append(n_rag)
                ydata.append(num_gen)
                zdata.append(alpha_rag)
                c.append('red')

                simulation_points_temp = []
                num_simulation_point = num_point_dict[dataset]
                sample_size_per_point = sample_size_dict[dataset]
                for idx_point in range(num_simulation_point):
                    test_list = random.sample(list(range(len(results))), sample_size_per_point)
                    new_res = np.mean(results[test_list])
                    simulation_points_temp.append(new_res)
                simulation_points_temp = np.array(simulation_points_temp)

                ax.scatter3D([n_rag] * len(simulation_points_temp), [num_gen] * len(simulation_points_temp), simulation_points_temp, c='gray', alpha=0.3, s=25)

        ax.scatter3D(xdata, ydata, zdata, c=c, marker='^', s=80)

        labelsize = 20
        fontsize = 30
        ax.tick_params(axis='x', labelsize=labelsize,pad=-3)
        ax.tick_params(axis='y', labelsize=labelsize,pad=-3)
        ax.tick_params(axis='z', labelsize=labelsize,pad=8)
        ax.set_xlabel(r'$N_{rag}$', fontsize=fontsize, labelpad=10.0)
        ax.set_ylabel(r'$\lambda_g$', fontsize=fontsize, labelpad=10.0)
        ax.set_zlabel(r'Risk', fontsize=fontsize, labelpad=25.0, rotation=90)

        plt.tight_layout()
        plt.savefig(f'./figs/{dataset}_{method2method[method]}_nrag_lambdag_conformal_risk.jpg',dpi=800)