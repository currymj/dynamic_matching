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

def make_matching_matrix(l_n, r_n):
    # should take lhs and rhs sizes

    lhs = list(range(l_n))
    rhs = list(range(l_n, l_n + r_n))

    # n_vars is 1 per possible edge?
    n_vars = len(lhs) * len(rhs)
    # n_constraints is 1 for each lhs, 1 for each rhs, 1 per edge?
    n_constraints = len(lhs) + len(rhs) + n_vars
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros((n_constraints))
    curr_idx = 0
    edge_idx = {}
    # get an index per edge
    for u in lhs:
        for v in rhs:
            edge_idx[(u, v)] = curr_idx
            curr_idx += 1
    # A has rows of 2n elements, followed by n^2 edges
    # A has cols of n^2 edges (so A @ x where x is edges)
    for u in lhs:
        for v in rhs:
            # for u, flip on coefficient for only its outgoing edges
            A[u, edge_idx[(u, v)]] = 1
            # for v, flip on coefficient for only its incoming edges
            A[v, edge_idx[(u, v)]] = 1
            # for the edge itself, flip on a single -1 at its point only (- point must be <= 0 i.e. point must be positive)
            A[len(lhs) + len(rhs) + edge_idx[(u, v)], edge_idx[(u, v)]] = -1

    # each element can have only 1 edge turned on in x
    for u in lhs:
        b[u] = 1
    for u in rhs:
        b[u] = 1

    return A, b

class History:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class CurrentElems:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

def ind_counts_to_longs(arrival_counts):
    # optimize later
    results = []
    for i in range(arrival_counts.shape[0]):
        for j in range(arrival_counts[i]):
            results.append(i)
    return torch.LongTensor(results)

def both_sides_history(type_arrival_rates, type_departure_probs, max_t):
    return History(generate_full_history(type_arrival_rates, type_departure_probs, max_t),
                   generate_full_history(type_arrival_rates, type_departure_probs, max_t))

def generate_full_history(type_arrival_rates, type_departure_probs, max_t):
    # an element is a list of (type, start_time, end_time)
    # too bad we don't have mutable namedtuples here, and it's probably not
    # worth creating a tiny class
    all_elems = []
    curr_elems = []
    for t in range(max_t):
        # departures
        next_currelems = []
        for i in range(len(curr_elems)):
            v = curr_elems[i]
            departing = np.random.rand() <= type_departure_probs[v[0]]
            if departing:
                v[2] = t
            else:
                next_currelems.append(v)
        curr_elems = next_currelems

        arrival_types = ind_counts_to_longs(np.random.poisson(lam=type_arrival_rates))
        arrivals = [[x, t, -1] for x in arrival_types]
        all_elems.extend(arrivals)
        curr_elems.extend(arrivals)

    for v in curr_elems:
        v[2] = max_t
    for v in all_elems:
        assert(v[1] >= 0)
        assert(v[2] >= 0)

    return all_elems

def toy_e_weights_type():
    mat = 0.1*torch.ones(5,5)
    mat[0,1] = 3.0
    mat[1,0] = 3.0
    mat[0,0] = -100.0
    mat[0,2:5] = -100.0
    mat[2:5,0] = -100.0
    return mat

toy_arrival_rates = torch.Tensor([0.2,1.0,1.0,1.0,1.0])
toy_departure_probs = torch.Tensor([0.9,0.05,0.1,0.1,0.1])

def get_matched_indices(match_edges, e_weights, match_thresh=0.8):
    lhs_matched_inds = []
    rhs_matched_inds = []
    total_true_loss = 0.0
    for i in range(match_edges.shape[0]):
        max_val, max_ind = torch.max(match_edges[i], 0)
        if max_val > match_thresh:
            lhs_matched_inds.append(i)
            rhs_matched_inds.append(max_ind.item())
            total_true_loss += e_weights[i, max_ind].item()

    return lhs_matched_inds, rhs_matched_inds, total_true_loss

def weight_matrix(lhs_current_elems, rhs_current_elems, weights_by_type):
    # optimize later
    weights_result = torch.zeros(lhs_current_elems.shape[0], rhs_current_elems.shape[0])
    for i in range(lhs_current_elems.shape[0]):
        for j in range(rhs_current_elems.shape[0]):
            weights_result[i,j] = weights_by_type[lhs_current_elems[i],rhs_current_elems[j]]
    return weights_result

def type_weight_matrix(lhs_current_elems, rhs_current_elems, weights_by_type):
    # optimize later
    weights_result = torch.zeros(lhs_current_elems.shape[0], rhs_current_elems.shape[0])
    for i in range(lhs_current_elems.shape[0]):
        for j in range(rhs_current_elems.shape[0]):
            weights_result[i, j] = weights_by_type[lhs_current_elems[i]] + weights_by_type[rhs_current_elems[j]]
    return weights_result

def compute_matching(current_pool_list, curr_type_weights, e_weights_by_type, gamma=0.000001):
    # current_pool_list should have lhs and rhs, get them both as tensors
    lhs_current_elems = torch.tensor([x[0] for x in current_pool_list.lhs])
    rhs_current_elems = torch.tensor([x[0] for x in current_pool_list.rhs])
    l_n = lhs_current_elems.shape[0]
    r_n = rhs_current_elems.shape[0]
    A, b = make_matching_matrix(l_n, r_n)
    A = torch.from_numpy(A).float()
    b = torch.from_numpy(b).float()
    # should take lhs and rhs
    e_weights = weight_matrix(lhs_current_elems, rhs_current_elems, e_weights_by_type).view(l_n, r_n)
    jitter_e_weights = e_weights + 1e-4 * torch.rand(l_n, r_n)
    # e_weights = torch.rand(n,n)
    model_params_quad = make_gurobi_model(A.detach().numpy(), b.detach().numpy(), None, None,
                                          gamma * np.eye(A.shape[1]))
    func = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad)

    Q_mat = gamma * torch.eye(A.shape[1])

    curr_elem_weights = type_weight_matrix(lhs_current_elems, rhs_current_elems, curr_type_weights).view(l_n, r_n)
    modified_edge_weights = jitter_e_weights - 0.5 * (curr_elem_weights)
    # may need some negative signs
    resulting_match = func(Q_mat, -modified_edge_weights.view(-1), A, b, torch.Tensor(), torch.Tensor()).view(l_n, r_n)
    return resulting_match, e_weights

