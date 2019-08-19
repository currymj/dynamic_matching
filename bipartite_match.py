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

