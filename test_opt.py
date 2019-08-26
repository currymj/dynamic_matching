from bipartite_match import History, CurrentElems, toy_e_weights_type
from opt import opt_match
import torch

def test_compute_opt_basic():
    hist = History([[torch.tensor(1), 0, 2]], [[torch.tensor(1), 0, 2]])
    max_t = 2
    e_weights_type = toy_e_weights_type()

    result_dict, model = opt_match(hist, max_t, e_weights_type)

    assert result_dict[0,0,0].x == 1.0

def test_compute_opt_tiebreak():
    pass
    # make sure the tiebreaking works correctly (it should bc binary vars explicitly)