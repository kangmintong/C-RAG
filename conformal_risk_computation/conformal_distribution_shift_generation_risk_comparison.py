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
from utils import save_json_to_file, get_method2name, get_method2method, get_num_points, cal_dist_shift_bound, get_color_dict, get_max_x_all

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute and plot the conformal generation risks of one retrieval model.')

    parser.add_argument('--llm_retrieval_methods', '--names-list', nargs='+')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--n_rag', type=int, default=15)
    parser.add_argument('--num_gen', type=int, default=1)
    parser.add_argument('--gen_thresh', type=int, default=0)

    args = parser.parse_args()

    method2name = get_method2name()
    method2method = get_method2method()

    datasets = args.datasets
    retrieval_methods = args.llm_retrieval_methods

    max_x_all = get_max_x_all()

    n_rag = args.n_rag
    num_gen = args.num_gen
    gen_thresh = args.gen_thresh

    for dataset in datasets:
        max_x = max_x_all[dataset]
        hellinger_distances = np.array(list(range(0, max_x, 1)))
        hellinger_distances = hellinger_distances / 100.0

        alphas = {}
        min_risk, max_risk = 10e5, -10e5

        for method in retrieval_methods:

            alpha_list = []
            evaluate_results = json.load(open(f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{num_gen}_{gen_thresh}_calibration_result.json','r', encoding='utf-8'))


            plot_dist_shift_results = {}


            results = list(evaluate_results.values())[0][0]
            results = np.array(results)
            r_hat = np.mean(results)
            r_hat = 1. - r_hat

            for dist in tqdm(hellinger_distances):

                temp_res = cal_dist_shift_bound(r_hat, dist, len(results), np.var(results), 0.1)
                plot_dist_shift_results[dist] = temp_res
                alpha_list.append(temp_res)

                if temp_res < min_risk:
                    min_risk = temp_res
                if temp_res > max_risk:
                    max_risk = temp_res

            alphas[method] = alpha_list

        data = {
            'dist': hellinger_distances,
        }
        for method in retrieval_methods:
            data[method] = alphas[method]

        df = pd.DataFrame(data)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10, 8))

        color_dict = get_color_dict(retrieval_methods)

        for metric, color in color_dict.items():
            plt.plot(df['dist'], df[metric], marker='*', color=color, label=method2name[metric], markersize=20)

        fontsize = 32
        plt.ylim([min_risk - 0.07, min(max_risk + 0.02, 1.02)])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter('{x:3<0.2f}')
        ax.yaxis.set_major_formatter('{x:3<0.2f}')

        # Show the plot
        plt.tight_layout()
        plt.savefig(f'./figs/{dataset}_distribution_shift_conformal_risk.jpg', dpi=800)