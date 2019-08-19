import torch

from bipartite_match import ind_counts_to_longs, get_matched_indices
from collections import defaultdict
from pytest import approx

def test_ind_counts_to_longs():
    arrival_counts = torch.tensor([2,3,2])
    result = ind_counts_to_longs(arrival_counts)
    result_set = set(result.numpy().tolist())
    assert result_set == set([0,0,1,1,1,2,2])


from bipartite_match import CurrentElems, toy_e_weights_type
from compute_matching import weight_matrix


def unambiguous_matching():
    currpool = CurrentElems([[torch.tensor(2), 0, 5], [torch.tensor(1), 0, 5], [torch.tensor(2), 0, 5]],
                            [[torch.tensor(0), 0, 5]])
    e_weights_type = toy_e_weights_type()
    e_weights = weight_matrix(currpool.lhs, currpool.rhs, e_weights_type)
    correct_matching = torch.tensor([[0.0],
                                     [1.0],
                                     [0.0]])
    return currpool, e_weights_type, e_weights, correct_matching


# testing index getting
def test_get_matched_indices_1():
    currpool, e_weights_type, e_weights, correct_matching = unambiguous_matching()
    lhs_inds, rhs_inds, _ = get_matched_indices(correct_matching, e_weights)
    assert set(lhs_inds) == set([1])
    assert set(rhs_inds) == set([0])

def test_empty_match():
    currpool, e_weights_type, e_weights, correct_matching = unambiguous_matching()
    lhs_inds, rhs_inds, _ = get_matched_indices(torch.zeros_like(correct_matching), e_weights)
    assert set(lhs_inds) == set([])
    assert set(rhs_inds) == set([])

def test_screwy_match():
    currpool, e_weights_type, e_weights, correct_matching = unambiguous_matching()
    correct_matching[0,0] = 0.1
    correct_matching[1,0] = 0.9
    lhs_inds, rhs_inds, _ = get_matched_indices(correct_matching, e_weights)
    assert set(lhs_inds) == set([1])
    assert set(rhs_inds) == set([0])


def test_match_value():
    currpool, e_weights_type, e_weights, correct_matching = unambiguous_matching()
    lhs_inds, rhs_inds, true_loss = get_matched_indices(correct_matching, e_weights)
    assert true_loss == 3.0

def test_empty_match_value():
    currpool, e_weights_type, e_weights, correct_matching = unambiguous_matching()
    lhs_inds, rhs_inds, true_loss = get_matched_indices(torch.zeros_like(correct_matching), e_weights)
    assert true_loss == 0.0

def test_screwy_match_value():
    currpool, e_weights_type, e_weights, correct_matching = unambiguous_matching()
    correct_matching[0,0] = 0.1
    correct_matching[1,0] = 0.9
    lhs_inds, rhs_inds, true_loss = get_matched_indices(correct_matching, e_weights)
    assert true_loss == 3.0


from bipartite_match import History, history_to_arrival_dict, arrivals_only, step_simulation

def test_history_to_arrival_dict():
    hist = History([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5], [torch.tensor(2), 2, 5]],
                 [[torch.tensor(1), 0, 1], [torch.tensor(1), 1, 5]])
    desired_dict_lhs = defaultdict(list)
    desired_dict_lhs[0] = [[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]]
    desired_dict_lhs[2] = [[torch.tensor(2), 2, 5]]

    desired_dict_rhs = defaultdict(list)
    desired_dict_rhs[0] = [[torch.tensor(1), 0, 1]]
    desired_dict_rhs[1] = [[torch.tensor(1), 1, 5]]

    assert history_to_arrival_dict(hist.lhs) == desired_dict_lhs
    assert history_to_arrival_dict(hist.rhs) == desired_dict_rhs

def test_arrivals_only():
    hist = History([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5], [torch.tensor(2), 2, 5]],
                   [[torch.tensor(1), 0, 1], [torch.tensor(1), 1, 5]])
    desired_dict_lhs = defaultdict(list)
    desired_dict_lhs[0] = [[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]]
    desired_dict_lhs[2] = [[torch.tensor(2), 2, 5]]

    desired_dict_rhs = defaultdict(list)
    desired_dict_rhs[0] = [[torch.tensor(1), 0, 1]]
    desired_dict_rhs[1] = [[torch.tensor(1), 1, 5]]

    currpool = CurrentElems([],[])
    newpool = arrivals_only(currpool, desired_dict_lhs, desired_dict_rhs, 0)

    target_pool = CurrentElems([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]],[[torch.tensor(1), 0, 1]])
    assert newpool.lhs == target_pool.lhs
    assert newpool.rhs == target_pool.rhs

def test_step_simulation_noarrival():
    desired_dict_lhs = defaultdict(list)
    desired_dict_rhs = defaultdict(list)


    currpool = CurrentElems([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]], [[torch.tensor(1), 0, 1]])
    match_edges = torch.tensor([[0.0],
                                [1.0]])
    e_weights_type = toy_e_weights_type()
    e_weights = weight_matrix(currpool.lhs, currpool.rhs, e_weights_type)

    result_pool, total_loss = step_simulation(currpool, match_edges, e_weights, desired_dict_lhs, desired_dict_rhs, 1)

    assert approx(total_loss, 0.1)

    assert result_pool.lhs == [[torch.tensor(1), 0, 2]]
    assert result_pool.rhs == []

def test_step_simulation_nomatch():
    desired_dict_lhs = defaultdict(list)
    desired_dict_rhs = defaultdict(list)


    currpool = CurrentElems([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]], [[torch.tensor(1), 0, 1]])
    match_edges = torch.tensor([[0.0],
                                [0.0]])
    e_weights_type = toy_e_weights_type()
    e_weights = weight_matrix(currpool.lhs, currpool.rhs, e_weights_type)

    result_pool, total_loss = step_simulation(currpool, match_edges, e_weights, desired_dict_lhs, desired_dict_rhs, 1)

    assert approx(total_loss, 0.0)

    assert result_pool.lhs == [[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]]
    assert result_pool.rhs == []


def test_step_simulation():
    hist = History([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5], [torch.tensor(2), 2, 5]],
                   [[torch.tensor(1), 0, 1], [torch.tensor(1), 1, 5]])
    desired_dict_lhs = defaultdict(list)
    desired_dict_lhs[0] = [[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]]
    desired_dict_lhs[2] = [[torch.tensor(2), 2, 5]]

    desired_dict_rhs = defaultdict(list)
    desired_dict_rhs[0] = [[torch.tensor(1), 0, 1]]
    desired_dict_rhs[1] = [[torch.tensor(1), 1, 5]]

    currpool = CurrentElems([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]],[[torch.tensor(1), 0, 1]])
    match_edges = torch.tensor([[0.0],
                                [1.0]])
    e_weights_type = toy_e_weights_type()
    e_weights = weight_matrix(currpool.lhs, currpool.rhs, e_weights_type)

    result_pool, total_loss = step_simulation(currpool, match_edges, e_weights, desired_dict_lhs, desired_dict_rhs, 1)

    assert approx(total_loss, 0.1)

    assert result_pool.lhs == [[torch.tensor(1), 0, 2]]
    assert result_pool.rhs == [[torch.tensor(1), 1, 5]]

from compute_matching import compute_matching
def test_e2e():
    hist = History([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5], [torch.tensor(2), 2, 5]],
                   [[torch.tensor(1), 0, 1], [torch.tensor(1), 1, 5]])
    desired_dict_lhs = defaultdict(list)
    desired_dict_lhs[0] = [[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]]
    desired_dict_lhs[2] = [[torch.tensor(2), 2, 5]]

    desired_dict_rhs = defaultdict(list)
    desired_dict_rhs[0] = [[torch.tensor(1), 0, 1]]
    desired_dict_rhs[1] = [[torch.tensor(1), 1, 5]]

    currpool = CurrentElems([[torch.tensor(1), 0, 2], [torch.tensor(2), 0, 5]],[[torch.tensor(1), 0, 1]])

    e_weights_type = toy_e_weights_type()
    potentials = torch.tensor([0.0,0.0,0.0,0.0,0.0])
    match_edges, e_weights_full = compute_matching(currpool, potentials, e_weights_type)


    result_pool, total_loss = step_simulation(currpool, match_edges, e_weights_full, desired_dict_lhs, desired_dict_rhs, 1)

    assert approx(total_loss, 0.1)

    assert result_pool.lhs == [[torch.tensor(2), 0, 5]]
    assert result_pool.rhs == [[torch.tensor(1), 1, 5]]
