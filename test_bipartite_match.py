import torch
from bipartite_match import ind_counts_to_longs
def test_ind_counts_to_longs():
    arrival_counts = torch.tensor([2,3,2])
    result = ind_counts_to_longs(arrival_counts)
    result_set = set(result.numpy().tolist())
    assert result_set == set([0,0,1,1,1,2,2])