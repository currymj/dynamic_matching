import random

def opt_match(history, max_t, e_weights_type):
    import gurobipy as gp
    import numpy as np

    l_n = len(history.lhs)
    r_n = len(history.rhs)
    model = gp.Model()
    model.params.OutputFlag = 0

    x = {}
    for i in range(l_n):
        for j in range(r_n):
            for t in range(max_t):
                x[i, j, t] = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f'x_{i}_{j}_{t}',
                )
    model.update()

    # constraint: each node matched once
    match_once_constraints = []
    for i in range(l_n):
        match_once_constraints.append(
            model.addConstr(gp.quicksum(x[i, j, t] for j in range(r_n) for t in range(max_t)) <= 1))
    for j in range(r_n):
        match_once_constraints.append(
            model.addConstr(gp.quicksum(x[i, j, t] for i in range(l_n) for t in range(max_t)) <= 1))

    # constraint: for each node, zero before it arrives and after it departs
    arrive_depart_constraints = []
    for i, node_info in enumerate(history.lhs):
        t_arrive = node_info[1]
        t_depart = node_info[2]
        # for t in range(t_depart, max_t): # or is it t_depart + 1???? this is important!
        arrive_depart_constraints.append(
            model.addConstr(gp.quicksum(x[i, j, t] for j in range(r_n) for t in range(0, t_arrive)) == 0))
        arrive_depart_constraints.append(
            model.addConstr(gp.quicksum(x[i, j, t] for j in range(r_n) for t in range(t_depart, max_t)) == 0))

    for j, node_info in enumerate(history.rhs):
        t_arrive = node_info[1]
        t_depart = node_info[2]
        # for t in range(t_depart, max_t): # or is it t_depart + 1???? this is important!
        arrive_depart_constraints.append(
            model.addConstr(gp.quicksum(x[i, j, t] for i in range(l_n) for t in range(0, t_arrive)) == 0))
        arrive_depart_constraints.append(
            model.addConstr(gp.quicksum(x[i, j, t] for i in range(l_n) for t in range(t_depart, max_t)) == 0))
    # we don't need an additional binaryness constraint because of variable type
    # create objective while computing weights for each edge
    obj = gp.LinExpr()
    varwise_edge_weights = {}
    for i, node_info_i in enumerate(history.lhs):
        for j, node_info_j in enumerate(history.rhs):
            #random_jitter = random.random() * 1e-4
            # should add both tiebreaking and a small time-based discount
            i_type = node_info_i[0]
            j_type = node_info_j[0]
            varwise_edge_weights[i, j] = e_weights_type[i_type, j_type].item()
            edge_weight = -e_weights_type[i_type, j_type]
            for t in range(max_t):
                obj += x[i, j, t] * edge_weight.item()
    model.setObjective(obj, gp.GRB.MINIMIZE)

    model.optimize()
    model.update()


    #total_positive_obj = 0.0
    #for i in range(l_n):
        #for j in range(r_n):
            #for t in range(max_t):
                #val = x[i, j, t].x
                #if val > 0.0:
                    #total_positive_obj += varwise_edge_weights[i, j]

    return x, model


