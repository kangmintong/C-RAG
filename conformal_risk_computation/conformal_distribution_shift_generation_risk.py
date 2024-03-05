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
from utils import save_json_to_file, get_method2name, get_method2method, get_num_points_distribution_shift, get_max_x_all, cal_dist_shift_bound, compute_hellinger_distance

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute and plot the conformal generation risks of one retrieval model.')

    parser.add_argument('--llm_retrievalmodel', type=str, choices=['llama-7b_BM25', 'llama-7b_BAAI', 'llama-7b_OpenAI', 'llama-7b_triever-base'])
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--n_rag', type=int, default=15)
    parser.add_argument('--num_gen', type=int, default=1)
    parser.add_argument('--gen_thresh', type=int, default=0)

    args = parser.parse_args()

    method2name = get_method2name()
    method2method = get_method2method()

    datasets = args.datasets
    method = args.llm_retrievalmodel

    num_point_dict, sample_size_dict, simulate_num = get_num_points_distribution_shift()
    max_x_all = get_max_x_all()

    delta = 0.1

    for dataset in datasets:
        n_rag = args.n_rag
        num_gen = args.num_gen
        gen_thresh = args.gen_thresh

        max_x = max_x_all[dataset]
        x_interval = 1

        hellinger_distances = np.array(list(range(0, max_x, x_interval)))
        hellinger_distances = hellinger_distances / 100.0

        num_simulation_point = num_point_dict[dataset]
        sample_size_per_point = sample_size_dict[dataset]

        alphas = {}
        min_risk, max_risk = 10e5, -10e5

        alpha_list = []
        path = f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{num_gen}_{gen_thresh}_calibration_result.json'
        results = json.load(open(path, 'r', encoding='utf-8'))
        results = list(results.values())[0][0]
        results = np.array(results)
        r_hat = np.mean(results)
        r_hat = 1. - r_hat

        print(f'empirical risk: {r_hat}')
        plot_dist_shift_results = {}

        for dist in tqdm(hellinger_distances):

            temp_res = cal_dist_shift_bound(r_hat, dist, len(results), np.var(results), delta)
            plot_dist_shift_results[dist] = temp_res

            print(f'Hellinger distance: {dist}, results: {temp_res}')

            alpha_list.append(temp_res)
            plot_dist_shift_results[dist] = temp_res
            if temp_res < min_risk:
                min_risk = temp_res
            if temp_res > max_risk:
                max_risk = temp_res

            if r_hat < min_risk:
                min_risk = r_hat
        print(alpha_list)
        alphas[method] = alpha_list

        data = {
            'dist': hellinger_distances,
             method: alphas[method],
        }
        df = pd.DataFrame(data)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10, 8))
        color_dict = {method: 'black'}

        for metric, color in color_dict.items():
            plt.plot(df['dist'], df[metric], marker='^', color=color, label=r'Certified Conformal Risk $\alpha_{\text{rag}}$', markersize=20)

        plt.scatter(x=[0.0], y=[r_hat], marker='s', color='orange', label=r'empirical risk without distribution shifts', s=500)

        # simulation
        simulation_points = []
        simulate_x = []
        simulate_y = []
        results = json.load(open(f'outputs/{method2method[method]}/{method}/{dataset}_{n_rag}_{num_gen}_{gen_thresh}_calibration_result.json', 'r', encoding='utf-8'))
        results = list(results.values())[0][0]
        results = np.array(results)
        results_all = 1. - results

        for tmp_idx in tqdm(range(simulate_num)):

            # unconditional random sampling
            # sampled_weights = np.random.multivariate_normal(ori_weights, sample_cov*np.random.uniform(0,1))
            # sampled_weights[sampled_weights<0.] = 0.
            test_list = random.sample(list(range(len(results_all))), sample_size_per_point)
            results = results_all[test_list]

            results_mean = np.mean(results)
            index_large_risk = results >= results_mean
            index_small_risk = results < results_mean

            results_66 = np.quantile(a=results, q=0.66)
            results_33 = np.quantile(a=results, q=0.33)
            index_large = results >= results_66
            index_med = (results < results_66) & (results >= results_33)
            index_small = results < results_33

            results_75 = np.quantile(a=results, q=0.75)
            results_50 = np.quantile(a=results, q=0.50)
            results_25 = np.quantile(a=results, q=0.25)
            index_1 = results >= results_75
            index_2 = (results < results_75) & (results >= results_50)
            index_3 = (results < results_50) & (results >= results_25)
            index_4 = results < results_25

            ori_weights = np.ones_like(results) / len(results)
            sample_cov = np.eye(len(results)) * 1e-5

            # conditional sampling
            sampled_weights = np.ones_like(ori_weights)
            if np.random.uniform(0, 1) < 1. / 3:
                temp = np.random.uniform(0, 1)
                sampled_weights[index_large_risk] = np.random.uniform(temp, 1.0)
                sampled_weights[index_small_risk] = np.random.uniform(0.0, temp)
            elif np.random.uniform(0, 1) < 2. / 3:
                temp = np.random.uniform(0, 1)
                temp2 = np.random.uniform(0, temp)
                sampled_weights[index_large] = np.random.uniform(temp, 1.0)
                sampled_weights[index_med] = np.random.uniform(temp2, temp)
                sampled_weights[index_small] = np.random.uniform(0.0, temp2)

                # temp = np.random.uniform(0.1, 0.9)
                # sampled_weights[index_large_risk] = np.random.uniform(temp, temp+0.1)
                # sampled_weights[index_small_risk] = np.random.uniform(temp-0.1, temp)
                # temp = np.random.uniform(0, 1)
                # # print(sampled_weights[index_large_risk].shape)
                # # print(np.random.multivariate_normal(np.ones(len(index_large_risk))*0.5, np.eye(len(index_large_risk))* 1e-5 *np.random.uniform(0,1)).shape)
                # sampled_weights[index_large_risk] = np.random.multivariate_normal(np.ones(np.sum(index_large_risk==True))*0.5, np.eye(np.sum(index_large_risk==True))* 1e-5 *np.random.uniform(0,1))
                # sampled_weights[index_small_risk] = np.random.multivariate_normal(np.ones(np.sum(index_small_risk==True))*2.0, np.eye(np.sum(index_small_risk==True))* 1e-5 *np.random.uniform(0,1))
            else:
                temp1 = np.random.uniform(0, 1)
                temp2 = np.random.uniform(0, temp1)
                temp3 = np.random.uniform(0, temp2)
                sampled_weights[index_1] = np.random.uniform(temp1, 1.0)
                sampled_weights[index_2] = np.random.uniform(temp2, temp1)
                sampled_weights[index_3] = np.random.uniform(temp3, temp2)
                sampled_weights[index_4] = np.random.uniform(0, temp3)

                # sampled_weights = np.random.multivariate_normal(ori_weights, sample_cov * np.random.uniform(0, 1))
            sampled_weights[sampled_weights < 0.] = 0.

            # normalize
            sampled_weights = sampled_weights / linalg.norm(sampled_weights, ord=1)

            dist = compute_hellinger_distance(ori_weights, sampled_weights)
            new_res = np.dot(sampled_weights, results)
            if dist > (max_x - 1) / 100.0:
                continue

            simulate_x.append(dist)
            simulate_y.append(new_res)

        simulate_x = np.array(simulate_x)
        simulate_y = np.array(simulate_y)

        for y in simulate_y:
            if y < min_risk:
                min_risk = y
            if y > max_risk:
                max_risk = y
        plt.scatter(x=simulate_x, y=simulate_y, color='gray', alpha=0.7, s=100)


        fontsize = 32
        plt.ylim([min_risk - 0.07, min(max_risk + 0.02, 1.02)])
        plt.xlim([-0.01, max_x / 100.0])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter('{x:9<0.2f}')
        ax.yaxis.set_major_formatter('{x:9<0.2f}')

        plt.tight_layout()
        print('save figure at {}'.format({f'./figs/{dataset}_{method2name[method]}_dist_shift_conformal_bound_simulation.jpg'}))
        plt.savefig(f'./figs/{dataset}_{method2name[method]}_dist_shift_conformal_bound_simulation.jpg', dpi=800)