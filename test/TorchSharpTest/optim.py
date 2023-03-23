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