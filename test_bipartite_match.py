import torch

from bipartite_match import ind_counts_to_longs, get_matched_indices

def test_ind_counts_to_longs():
    arrival_counts = torch.tensor([2,3,2])
    result = ind_counts_to_longs(arrival_counts)
    result_set = set(result.numpy().tolist())
    assert result_set == set([0,0,1,1,1,2,2])


from bipartite_match import CurrentElems, toy_e_weights_type

def unambiguous_matching():
    currpool = CurrentElems([[torch.tensor(1), 0, 5], [torch.tensor(2), 0, 5], [torch.tensor(2), 0, 5]],
                            [[torch.tensor(0), 0, 5]])
    e_weights = toy_e_weights_type()
    correct_matching = torch.tensor([[1.0],
                                     [0.0],
                                     [0.0]])
    return currpool, e_weights, correct_matching


def test_get_matched_indices_1():
    currpool, e_weights, correct_matching = unambiguous_matching()
    lhs_inds, rhs_inds, _ = get_matched_indices(correct_matching, e_weights)
    assert set(lhs_inds) == set([0])
    assert set(rhs_inds) == set([0])

def test_empty_match():
    currpool, e_weights, correct_matching = unambiguous_matching()
    lhs_inds, rhs_inds, _ = get_matched_indices(torch.zeros_like(correct_matching), e_weights)
    assert set(lhs_inds) == set([])
    assert set(rhs_inds) == set([])

def test_screwy_match():
    currpool, e_weights, correct_matching = unambiguous_matching()
    correct_matching[0,0] = 0.9
    correct_matching[1,0] = 0.1
    lhs_inds, rhs_inds, _ = get_matched_indices(correct_matching, e_weights)
    assert set(lhs_inds) == set([0])
    assert set(rhs_inds) == set([0])




