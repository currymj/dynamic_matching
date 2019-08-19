import torch
from bipartite_match import CurrentElems, toy_e_weights_type
from compute_matching import compute_matching

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

    resulting_match, e_weights = compute_matching(currpool, torch.zeros(5), e_weights_type)
    assert torch.allclose(resulting_match, correct_matching)