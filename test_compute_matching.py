import torch
from bipartite_match import CurrentElems, toy_e_weights_type
from compute_matching import compute_matching, jitter_matrix, weight_matrix

def unambiguous_matching():
    currpool = CurrentElems([[torch.tensor(2), 0, 5], [torch.tensor(1), 0, 5], [torch.tensor(2), 0, 5]],
                            [[torch.tensor(0), 0, 5]])
    e_weights = toy_e_weights_type()
    correct_matching = torch.tensor([[0.0],
                                     [1.0],
                                     [0.0]])
    return currpool, e_weights, correct_matching

def test_compute_matching_noweights():
    currpool, e_weights_type, correct_matching = unambiguous_matching()
    e_weights_full = weight_matrix(currpool.lhs, currpool.rhs, e_weights_type)

    resulting_match, e_weights = compute_matching(currpool, torch.zeros(5), e_weights_full)
    assert torch.allclose(resulting_match, correct_matching)

def ambiguous_matching():
    currpool = CurrentElems([[torch.tensor(2), 0, 5], [torch.tensor(1), 0, 5], [torch.tensor(2), 0, 5]],
                            [[torch.tensor(0), 0, 5], [torch.tensor(0), 0, 5]])


    e_weights = toy_e_weights_type()
    one_correct_matching = torch.tensor([[0.0, 0.0],
                                     [1.0, 0.0],
                                     [0.0, 0.0]])
    return currpool, e_weights, one_correct_matching

def test_compute_ambiguous():
    # ties ought to be broken deterministically, by ordering (should jitter edge weights in this way)
    currpool, e_weights_type, one_correct_matching = ambiguous_matching()
    e_weights_full = weight_matrix(currpool.lhs, currpool.rhs, e_weights_type)

    resulting_match, e_weights = compute_matching(currpool, torch.zeros(5), e_weights_full)
    assert torch.allclose(resulting_match, one_correct_matching, atol=1e-6)


def test_jitter_matrix():
    jit_mat = jitter_matrix(2,4)
    assert torch.allclose(jit_mat, torch.tensor([[2.0,1.666,1.333,1.0],
                                                 [1.0,0.666,0.333,0.0]]), atol=1e-3)

def test_weight_matrix():
    currpool = CurrentElems([[torch.tensor(2), 0, 5], [torch.tensor(1), 0, 5], [torch.tensor(2), 0, 5]],
                            [[torch.tensor(0), 0, 5]])
    e_weights = toy_e_weights_type()
    result_weights = weight_matrix(currpool.lhs, currpool.rhs, e_weights)
    assert torch.allclose(result_weights, torch.tensor([[-100.0],
                           [3.0],
                           [-100.0]]))
