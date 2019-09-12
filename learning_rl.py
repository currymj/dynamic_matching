import torch
import torch.nn as nn
import random
import numpy as np
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from collections import defaultdict

from bipartite_match import *
from compute_matching import compute_matching

def compute_discounted_returns(losses, gamma=1.0):
    # inspired originally by facebook's reinforce example
    returns = []
    R = 0.0
    for r in losses[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def sample_from_match(match_matrix):
    row_len, col_len = match_matrix.shape
    excluded_cols = set([])
    sampled_edges = []
    joint_log_prob = 0.0
    for row in range(row_len):
        for col in range(col_len):
            if col in excluded_cols:
                continue
            this_edge_prob = match_matrix[row, col]
            # flip coin here to include edge
            edge_include = random.random() <= this_edge_prob
            if edge_include:
                joint_log_prob += torch.log(this_edge_prob)
                excluded_cols.add(col)
                sampled_edges.append((row, col))
                break

    return sampled_edges, joint_log_prob

def train_func(list_of_histories, n_rounds=50, n_epochs=20):
    e_weights_type = toy_e_weights_type()
    type_weights = torch.full((5,), 0.0, requires_grad=True)
    optimizer = torch.optim.Adam([type_weights], lr=2e-3, weight_decay=1e-1)
    total_losses = []
    for e in tqdm(range(n_epochs)):
        full_history = list_of_histories[e]
        l_t_to_arrivals = history_to_arrival_dict(full_history.lhs)
        r_t_to_arrivals = history_to_arrival_dict(full_history.rhs)
        optimizer.zero_grad()
        rewards = []
        logprobs = []
        curr_pool = CurrentElems([], [])
        for r in range(n_rounds):
            if len(curr_pool.lhs) <= 0 or len(curr_pool.rhs) <= 0:
                curr_pool = arrivals_only(curr_pool, l_t_to_arrivals, r_t_to_arrivals, r)
                continue
            match_matrix, e_weights = compute_matching(curr_pool, type_weights, e_weights_type)
            resulting_match, match_logprob = sample_from_match(match_matrix)
            # resulting_match here should be a list of tuples
            curr_pool, true_loss = step_simulation_sampled(curr_pool, resulting_match, e_weights, l_t_to_arrivals,
                                                   r_t_to_arrivals, r)
            rewards.append(true_loss)
            logprobs.append(match_logprob)
        disc_returns = compute_discounted_returns(rewards)
        total_loss = torch.sum(torch.stack(-disc_returns * logprobs))
        total_losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()
    return type_weights, total_losses

def eval_func(list_of_histories, trained_weights, n_rounds = 50, n_epochs=100):
    e_weights_type = toy_e_weights_type()
    type_weights = trained_weights.detach()
    all_losses = []
    for e in tqdm(range(n_epochs)):
        full_history = list_of_histories[e]
        l_t_to_arrivals = history_to_arrival_dict(full_history.lhs)
        r_t_to_arrivals = history_to_arrival_dict(full_history.rhs)
        losses = []
        curr_pool = CurrentElems([], [])
        for r in range(n_rounds):
            if len(curr_pool.lhs) <= 0 or len(curr_pool.rhs) <= 0:
                curr_pool = arrivals_only(curr_pool, l_t_to_arrivals, r_t_to_arrivals, r)
                losses.append(0.0)
                continue
            match_matrix, e_weights = compute_matching(curr_pool, type_weights, e_weights_type)
            #losses.append(1.0*torch.sum(resulting_match * e_weights).item())
            #old_loss = 1.0*torch.sum(resulting_match * e_weights).item()
            #new_loss = 1.0*true_match_loss(resulting_match, e_weights)
            #if abs(new_loss - old_loss) > 0.1:
            #print(f'old - new: {old_loss - new_loss}')
            #print(resulting_match)
            #losses.append(1.0*true_match_loss(resulting_match, e_weights))
            resulting_match, _ = sample_from_match(match_matrix)
            curr_pool, true_loss = step_simulation_sampled(curr_pool, resulting_match, e_weights, l_t_to_arrivals, r_t_to_arrivals, r)
            losses.append(true_loss)
        if len(losses) == 0:
            losses.append(0.0)
        all_losses.append(losses)
    return all_losses


if __name__ == '__main__':
    results_list = []
    train_epochs = 50
    test_epochs = 100
    n_experiments = 3
    n_rounds = 50
    edge_weights = toy_e_weights_type()

    for i in range(n_experiments):
        print(i)
        print('generating histories for training')
        list_of_histories = [both_sides_history(toy_arrival_rates, toy_departure_probs, n_rounds) for e in
                             tqdm(range(train_epochs))]
        result_weights, learning_loss = train_func(list_of_histories, n_epochs=train_epochs)
        print(result_weights)
        print('generating histories for testing')
        test_histories = [both_sides_history(toy_arrival_rates, toy_departure_probs, n_rounds) for e in
                          tqdm(range(test_epochs))]
        loss_list = eval_func(test_histories, result_weights, n_epochs=test_epochs)
        learned_loss = np.mean([np.sum(l) for l in loss_list])
        learned_std = np.std([np.sum(l) for l in loss_list])
        print('loss of learned weights:', learned_loss)
        print('std of learned weights:', learned_std)

        const_loss_list = eval_func(test_histories, torch.full((5,), 0.0, requires_grad=False), n_epochs=test_epochs)
        const_loss = np.mean([np.sum(l) for l in const_loss_list])
        const_std = np.std([np.sum(l) for l in const_loss_list])
        print('loss of initial constant weights:', const_loss)
        print('std of initial constant weights:', const_std)

        # print('computing OPT scores')
        # optimal_loss_list = [opt_score(h, n_rounds, edge_weights) for h in tqdm(test_histories)]

        # learned_regret = [(optimal - l) for (l, optimal) in zip([np.sum(l) for l in loss_list], optimal_loss_list)]
        learned_regret = [0.0]
        learned_regret_mean = np.mean(learned_regret)
        learned_regret_std = np.std(learned_regret)

        # const_regret = [(optimal - l) for (l, optimal) in zip([np.sum(l) for l in const_loss_list], optimal_loss_list)]
        const_regret = [0.0]
        const_regret_mean = np.mean(const_regret)
        const_regret_std = np.std(const_regret)

        results_list.append((learned_loss, learned_std, const_loss, const_std, learned_regret_mean, learned_regret_std,
                             const_regret_mean, const_regret_std))

    for i in range(n_experiments):
        print('experiment', i)
        losses = results_list[i]
        learned_ci = 1.96 * losses[1] / np.sqrt(test_epochs)
        learned_r_ci = 1.96 * losses[5] / np.sqrt(test_epochs)
        const_ci = 1.96 * losses[3] / np.sqrt(test_epochs)
        const_r_ci = 1.96 * losses[7] / np.sqrt(test_epochs)

        print(f"learned weights mean: {losses[0]} +/- {learned_ci}")
        print(f"constant weights mean: {losses[2]} +/- {const_ci}")
        print(f'learned mean regret: {losses[4]} +/- {learned_r_ci}')
        print(f'constant weights mean regret: {losses[6]} +/- {const_r_ci}')
