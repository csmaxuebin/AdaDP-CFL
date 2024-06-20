import numpy as np
import torch
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def L2(old_params, old_w, param, args, rel):
    w = []
    for p in param:
        w.append(p.to(device))
    for i in range(len(w)):
        w[i] = torch.flatten(w[i])
    for i in range(len(old_w)):
        old_w[i] = torch.flatten(old_w[i].to(device))
    for i in range(len(old_params)):
        for j in range(len(old_params[i])):
            old_params[i][j] = torch.flatten(old_params[i][j].to(device))

    _w = w[0]
    for i in range(1, len(w)):
        _w = torch.cat([_w, w[i]])
    _old_w = old_w[0]
    for i in range(1, len(old_w)):
        _old_w = torch.cat([_old_w, old_w[i]])
    _old_params = []
    for i in range(len(old_params)):
        _old_param = old_params[i][0]
        for j in range(1, len(old_params[i])):
            _old_param = torch.cat([_old_param, old_params[i][j]])
        _old_params.append(_old_param)

    x = torch.sub(_w, _old_w)
    x = torch.norm(x, 'fro')
    x = torch.pow(x, 2)
    loss = x

    for i in range(len(_old_params)):
        _param = _old_params[i]
        x = torch.sub(_w, _param)
        x = torch.linalg.norm(x)
        x = torch.pow(x, 2)
        x = torch.mul(x, args.L)
        x = torch.mul(x, rel[i])
        loss = torch.add(loss, x)

    return loss







