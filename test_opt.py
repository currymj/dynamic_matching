from bipartite_match import History, CurrentElems, toy_e_weights_type
from opt import opt_match
import torch

def test_compute_opt_basic():
    hist = History([[torch.tensor(1), 0, 2]], [[torch.tensor(1), 0, 2]])
    max_t = 2
    e_weights_type = toy_e_weights_type()

    result_dict, model = opt_match(hist, max_t, e_weights_type)

    assert result_dict[0,0,0].x == 1.0

def test_compute_opt_multi():
    hist = History([[torch.tensor(1), 0, 2], [torch.tensor(1), 3, 4]], [[torch.tensor(1), 0, 2], [torch.tensor(1), 3, 4]])
    max_t = 4
    e_weights_type = toy_e_weights_type()

    result_dict, model = opt_match(hist, max_t, e_weights_type)

    assert result_dict[0,0,0].x == 1.0

    assert result_dict[1,1,3].x == 1.0

def test_compute_opt_smart():
    hist = History([[torch.tensor(1), 0, 4], [torch.tensor(2), 3, 4]], [[torch.tensor(1), 0, 4], [torch.tensor(0), 3, 4]])
    max_t = 4
    e_weights_type = toy_e_weights_type()

    result_dict, model = opt_match(hist, max_t, e_weights_type)

    assert result_dict[0,0,0].x == 0.0

    assert result_dict[1,1,3].x == 0.0

    assert result_dict[0,1,3].x == 1.0
    assert result_dict[1,0,3].x == 1.0


# do we need tiebreaking or discounting? in this case we don't necessarily care what the optimal solution is, just its best score
# determinism of course makes testing easier, but maybe that's not worth messing with the formulation of the problem

