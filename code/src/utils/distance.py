import copy
import math
import numpy as np
import torch


def Distance(args, clients, clusters):
    X = [[] for i in range(args.num_users)]
    w_locals1 = []
    for i in range(args.num_users):
        w_locals1.append(copy.deepcopy(clients[i].get_state_dict()))
    for i in range(args.num_users):
        for j in w_locals1[i].keys():
             X[i] += torch.flatten(w_locals1[i][j]).tolist()
    X = np.array(X)

    X_groups = []
    for i in range(len(clusters)):
        X_local = []
        for j in range(args.num_users):
            if j in clusters[i]:
                X_local.append(X[j])

        X_group = X_local[0]
        for j in range(1, len(X_local)):
            X_group += X_local[j]
        X_group /= len(X_local)
        X_groups.append(copy.deepcopy(X_group))

    rel = []
    for i in range(len(X_groups)):
        rel.append([])
        for j in range(len(X_groups)):
            if j != i:
                if args.dist == 'L2':
                    dist = np.linalg.norm(X_groups[i] - X_groups[j])
                    rel[-1].append(math.exp(-1 * dist))
                if args.dist == 'Equal':
                    rel[-1].append(0.5)
                if args.dist == 'L1':
                    dist = np.sum(np.abs(X_groups[i] - X_groups[j]))
                    rel[-1].append(math.exp(-1 * dist))
                if args.dist == 'cos':
                    dist = 1 - np.dot(X_groups[i], X_groups[j].T) / (a * b)
                    rel[-1].append(dist)
    return rel