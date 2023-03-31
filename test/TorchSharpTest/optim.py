# Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#
# This file is used to regenerate the optimizer state_dict test files,
# where PyTorch optimizer state is transferred to TorchSharp.
#
# Execute this from the command line in the $/tests/TorchSharpTest directory.


import torch
import shutil

shutil.copy2('../../src/Python/exportsd.py', 'exportsd.py')

import exportsd

lin1 = torch.nn.Linear(10,10)
lin2 = torch.nn.Linear(10,10)

input = torch.rand(4,10)

# SGD

optim = torch.optim.SGD(lin1.parameters(), lr=0.001, momentum=0.1)

output = lin1(input).sum()
output.backward()

optim.step()

f = open("sgd1.dat", "wb")
exportsd.save_sgd(optim, f)
f.close()

# ASGD

optim = torch.optim.ASGD(lin1.parameters(), lr=0.001, alpha=0.65, t0=1e5, lambd=1e-3)

output = lin1(input).sum()
output.backward()

optim.step()

f = open("asgd1.dat", "wb")
exportsd.save_asgd(optim, f)
f.close()


# RMSprop

seq = torch.nn.Sequential(lin1, lin2)


optim = torch.optim.RMSprop(lin1.parameters(), lr=0.001, momentum=0.1)
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'momentum' : 0, 'centered': True})

output = seq(input).sum()
output.backward()

optim.step()

f = open("rmsprop1.dat", "wb")
exportsd.save_rmsprop(optim, f)
f.close()

# Rprop

optim = torch.optim.Rprop(lin1.parameters(), lr=0.001, etas=(0.35, 1.5), step_sizes=(1e-5, 5))
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'etas': (0.45, 1.5), 'step_sizes': (1e-5, 5), 'maximize': True})

output = seq(input).sum()
output.backward()

optim.step()

f = open("rprop1.dat", "wb")
exportsd.save_rprop(optim, f)
f.close()

# Adam

optim = torch.optim.Adam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9))
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'betas' : (0.7, 0.79), 'amsgrad': True})

output = seq(input).sum()
output.backward()

optim.step()

f = open("adam1.dat", "wb")
exportsd.save_adam(optim, f)
f.close()

# AdamW

optim = torch.optim.AdamW(lin1.parameters(), lr=0.001, betas=(0.8, 0.9))
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'betas' : (0.7, 0.79), 'amsgrad': True})

output = seq(input).sum()
output.backward()

optim.step()

f = open("adamw1.dat", "wb")
exportsd.save_adamw(optim, f)
f.close()

# NAdam

optim = torch.optim.NAdam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9))
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'betas' : (0.7, 0.79), 'weight_decay': 0.3})

output = seq(input).sum()
output.backward()

optim.step()

f = open("nadam1.dat", "wb")
exportsd.save_nadam(optim, f)
f.close()

# RAdam

optim = torch.optim.RAdam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9))
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'betas' : (0.7, 0.79), 'weight_decay': 0.3})

output = seq(input).sum()
output.backward()

optim.step()

f = open("radam1.dat", "wb")
exportsd.save_radam(optim, f)
f.close()

# Adamax

optim = torch.optim.Adamax(lin1.parameters(), lr=0.001, betas=(0.8, 0.9))
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'betas' : (0.7, 0.79), 'weight_decay' : 0.3})

output = seq(input).sum()
output.backward()

optim.step()

f = open("adamax1.dat", "wb")
exportsd.save_adamax(optim, f)
f.close()

# Adadelta

optim = torch.optim.Adadelta(lin1.parameters(), lr=0.001, rho=0.85, weight_decay=0.3)
optim.add_param_group({'params': lin2.parameters(), 'lr': 0.01, 'rho' : 0.79, 'maximize': True})

output = seq(input).sum()
output.backward()

optim.step()

f = open("adadelta1.dat", "wb")
exportsd.save_adadelta(optim, f)
f.close()

# Adagrad

optim = torch.optim.Adagrad(lin1.parameters(), lr=0.001, lr_decay=0.85, weight_decay=0.3)

output = seq(input).sum()
output.backward()

optim.step()

f = open("adagrad1.dat", "wb")
exportsd.save_adagrad(optim, f)
f.close()