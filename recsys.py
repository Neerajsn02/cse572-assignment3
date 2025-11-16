from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import KFold, cross_validate
from surprise.accuracy import mae, rmse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
data = Dataset.load_from_file("ratings_small.csv", reader=reader)

# evaluation fucntion - run the chosen algorithm
def evaluate(algo, data, num_folds = 5):
    results = cross_validate(algo, data, measures=["MAE", "RMSE"], cv=num_folds, verbose=False)
    mae = np.mean(results['test_mae']) # average mae
    rmse = np.mean(results['test_rmse']) # average rmse

    return mae, rmse


# note - from docs, pmf is just svd with biased=False
def baseline_alg_eval(data):
    print(f"Running PMF (SVD) with 5 fold CV\n")
    pmf = SVD(biased=False) # this is pmf
    pmf_mae, pmf_rmse = evaluate(pmf, data, num_folds=5)

    print(f"Running User-based CF with 5 fold CV")
    sim_options_ub = {"user_based": True} # for user based cf
    ub_filter_algo = KNNBasic(sim_options=sim_options_ub,verbose=False)
    ub_mae, ub_rmse = evaluate(ub_filter_algo, data, num_folds=5)

    print(f"Running Item-based CF with 5 fold CV")
    sim_options_ib = {"user_based": False} # for user based cf
    ib_filter_algo = KNNBasic(sim_options=sim_options_ib,verbose=False)
    ib_mae, ib_rmse = evaluate(ib_filter_algo, data, num_folds=5)


    return pmf_mae, pmf_rmse, ub_mae, ub_rmse, ib_mae, ib_rmse


# These 4 lines run the baseline algorithms
pmf_mae, pmf_rmse, ub_mae, ub_rmse, ib_mae, ib_rmse = baseline_alg_eval(data)
print(f"the pmf mae = {pmf_mae} and pmf rmse = {pmf_rmse}") # the pmf mae = 0.7788136669676791 and pmf rmse = 1.0086160320646873
print(f"the ub mae = {ub_mae} and the ub rmse = {ub_rmse}") # the ub mae = 0.7438767060026268 and the ub rmse = 0.9677758739202116
print(f"the ib mae = {ib_mae} and the ib rmse = {ib_rmse}") # the ib mae = 0.7208216727285663 and the ib rmse = 0.934414211487711


# understand how similarity emasures affect mean mae and rmse of the models
sim_measures = ['cosine', 'msd', 'pearson']

def similarity_eval(sim_measures, data):
    # return a list of 3 dictionaries for the 3 sim measures
    results = []
    for sim in sim_measures:

        sim_options_ub = {'name': sim, 'user_based': True}
        algo_ub = KNNBasic(sim_options=sim_options_ub, verbose=False)
        ub_mae_sim, ub_rmse_sim = evaluate(algo_ub, data, num_folds=5)
        print(f"User-based with sim={sim}: MAE = {ub_mae_sim} and RMSE = {ub_rmse_sim}")

        sim_options_ib = {'name': sim, 'user_based': False}
        algo_ib = KNNBasic(sim_options=sim_options_ib, verbose=False)
        ib_mae_sim, ib_rmse_sim = evaluate(algo_ib, data, num_folds=5)
        print(f"Item-based with sim={sim}: MAE = {ib_mae_sim} and RMSE = {ib_rmse_sim}")

        results.append(
            {
                'sim': sim,
                'ub_mae': ub_mae_sim,
                'ub_rmse': ub_rmse_sim,
                'ib_mae': ib_mae_sim,
                'ib_rmse': ib_rmse_sim
            }
        )

    return results

def sim_plot(results):
    # this function plots 2 graphs, comparing similarity metric to MAE and RMSE
    sims = [s['sim'] for s in results]
    ub_mae_scores = [s['ub_mae'] for s in results]
    ub_rmse_scores = [s['ub_rmse'] for s in results]
    ib_mae_scores = [s['ib_mae'] for s in results]
    ib_rmse_scores = [s['ib_rmse'] for s in results]

    plt.figure()
    plt.title('MAE for Similarity Metrics')
    plt.plot(sims, ub_mae_scores, marker='s', label="User-based MAE")
    plt.plot(sims, ib_mae_scores, marker='s', label="Item-based MAE")
    plt.xlabel('Similarity Metric')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_sim.png')
    plt.close()

    plt.figure()
    plt.title('RMSE for Similarity Metrics')
    plt.plot(sims, ub_rmse_scores, marker='o', label="User-based RMSE")
    plt.plot(sims, ib_rmse_scores, marker='o', label="Item-based RMSE")
    plt.xlabel('Similarity Metric')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_sim.png')
    plt.close()

# these 2 lines compute the RMSE and MAE for each similarity measures, and generates the plots
results = similarity_eval(sim_measures, data)
sim_plot(results)

# Note - For the next section, we use MSD as the similarity measure
# examining number of neighbors

k_vals = list(range(10, 101, 10)) # k values will be 10, 20, 30, .., 90, 100

def kn_eval(k_vals, data):
    user_rmse_k = []
    item_rmse_k = []
    user_mae_k = []
    item_mae_k = []
    for k in k_vals:
        # user based CF
        sim_options_ub = {'name': 'msd', 'user_based': True}
        ub_algo_k = KNNBasic(k=k, sim_options=sim_options_ub, verbose=False)
        ub_mae, ub_rmse = evaluate(ub_algo_k, data, num_folds=5)
        user_mae_k.append(ub_mae)
        user_rmse_k.append(ub_rmse)
        print(f"For k={k}, user based mae={ub_mae} and user based rmse={ub_rmse}")

        # item based CF
        sim_options_ib = {'name': 'msd', 'user_based': False}
        ib_algo_k = KNNBasic(k=k, sim_options=sim_options_ib,verbose=False)
        ib_mae, ib_rmse = evaluate(ib_algo_k, data, num_folds=5)
        item_mae_k.append(ib_mae)
        item_rmse_k.append(ib_rmse)
        print(f"For k={k}, item based mae={ib_mae} and item based rmse={ib_rmse}")

    return user_mae_k, user_rmse_k, item_mae_k, item_rmse_k

def k_plot(user_mae_k, user_rmse_k, item_mae_k, item_rmse_k):
    
    plt.figure()
    plt.title("RMSE by number of neighbours")
    plt.plot(k_vals, user_rmse_k, marker='o', label="User Based CF")
    plt.plot(k_vals, item_rmse_k, marker='o', label="Item Based CF")
    plt.xlabel("K Neighbours")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.savefig('k_rmse.png')
    plt.close()

    plt.figure()
    plt.title("MAE by number of neighbours")
    plt.plot(k_vals, user_mae_k, marker='o', label="User Based CF")
    plt.plot(k_vals, item_mae_k, marker='o', label="Item Based CF")
    plt.xlabel("K Neighbours")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig('k_mae.png')
    plt.close()


user_mae_k, user_rmse_k, item_mae_k, item_rmse_k = kn_eval(k_vals, data)
k_plot(user_mae_k, user_rmse_k, item_mae_k, item_rmse_k)

# Finding best k
def best_k(user_rmse_k, item_rmse_k):
    min_val_ub = min(user_rmse_k)
    min_index_ub = user_rmse_k.index(min_val_ub)
    min_val_ib = min(item_rmse_k)
    min_index_ib = item_rmse_k.index(min_val_ib)
    min_index_ub = (min_index_ub + 1) * 10
    min_index_ib = (min_index_ib + 1) * 10
    # index 1 => k = 10 ; index 2 => k = 20 ; ...
    return min_index_ub, min_val_ub, min_index_ib, min_val_ib


min_index_ub, min_val_ub, min_index_ib, min_val_ib = best_k(user_rmse_k, item_rmse_k)
print(f"Best value of K for user based CF = {min_index_ub}, with RMSE={min_val_ub}")
print(f"Best value of K for Item based CF = {min_index_ib}, with RMSE={min_val_ib}")

