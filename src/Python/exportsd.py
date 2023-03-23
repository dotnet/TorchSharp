#
# Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#
import io
import torch
import leb128
import numpy as np

def _elem_type(t):
    dt = t.dtype

    if dt == torch.uint8:
        return 0
    elif dt == torch.int8:
        return 1
    elif dt == torch.int16:
        return 2
    elif dt == torch.int32:
        return 3
    elif dt == torch.int64:
        return 4
    elif dt == torch.float16:
        return 5
    elif dt == torch.float32:
        return 6
    elif dt == torch.float64:
        return 7
    elif dt == torch.bool:
        return 11
    elif dt == torch.bfloat16:
        return 15
    else:
        return 4711

def _write_tensor(t, stream):
    stream.write(leb128.u.encode(_elem_type(t)))
    stream.write(leb128.u.encode(len(t.shape)))
    for s in t.shape:
        stream.write(leb128.u.encode(s))
    stream.write(t.numpy().tobytes())

def save_state_dict(sd, stream):
    """
    Saves a PyToch state dictionary using the format that TorchSharp can
    read.

    :param sd: A dictionary produced by 'model.state_dict()'
    :param stream: An write stream opened for binary I/O.
    """
    stream.write(leb128.u.encode(len(sd)))
    for entry in sd:
        stream.write(leb128.u.encode(len(entry)))
        stream.write(bytes(entry, 'utf-8'))
        _write_tensor(sd[entry], stream)

def _write_encoded_int64(value, stream):
    stream.write(leb128.u.encode(value))

def _write_optim_name(name, stream):
    _write_encoded_int64(len(name),stream)
    stream.write(bytes(name, 'utf-8'))

def _write_bool(value, stream):
    stream.write(value.to_bytes(1, 'little'))

def _write_int32(value, stream):
    stream.write(value.to_bytes(4, 'little'))

def _write_int64(value, stream):
    stream.write(value.to_bytes(8, 'little'))

def _write_conditional_state_tensor(name, state, stream):
    if name not in state:
        _write_bool(False, stream)
    else:                
        buf = state[name]
        if buf == None:
            _write_bool(False, stream)
        else:
            _write_bool(True, stream)
            _write_tensor(buf, stream)

def save_sgd(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('SGD', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(5,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        floats[2] = pg['momentum']
        floats[3] = pg['dampening']
        floats[4] = pg['weight_decay']
        stream.write(floats.tobytes())
        _write_bool(pg['nesterov'], stream)
        _write_bool(pg['maximize'], stream)

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_conditional_state_tensor('momentum_buffer', st, stream)           

def save_asgd(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('ASGD', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(2,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        stream.write(floats.tobytes())
        _write_bool(pg['maximize'], stream)
        floats = np.empty(4,dtype=np.float64)
        floats[0] = pg['lambd']
        floats[1] = pg['alpha']
        floats[2] = pg['weight_decay']
        floats[3] = pg['t0']
        stream.write(floats.tobytes())

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            floats = np.empty(2,dtype=np.float64)
            floats[0] = st['eta']       
            floats[1] = st['mu']
            stream.write(floats.tobytes())
            _write_tensor(st['ax'], stream)

def save_rprop(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('Rprop', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(2,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        stream.write(floats.tobytes())
        _write_bool(pg['maximize'], stream)
        floats = np.empty(4,dtype=np.float64)
        etaminus, etaplus = pg['etas']
        min_step, max_step = pg['step_sizes']
        floats[0] = etaminus
        floats[1] = etaplus
        floats[2] = min_step
        floats[3] = max_step
        stream.write(floats.tobytes())

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(st["step"], stream)
            _write_tensor(st['prev'], stream)
            _write_tensor(st['step_size'], stream)

def save_rmsprop(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('RMSProp', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(2,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        stream.write(floats.tobytes())
        _write_bool(pg['maximize'], stream)
        floats = np.empty(4,dtype=np.float64)
        floats[0] = pg['momentum']
        floats[1] = pg['alpha']
        floats[2] = pg['eps']
        floats[3] = pg['weight_decay']
        stream.write(floats.tobytes())
        _write_bool(pg['centered'], stream)

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(st["step"], stream)
            _write_tensor(st['square_avg'], stream)
            _write_conditional_state_tensor('momentum_buffer', st, stream)
            _write_conditional_state_tensor('grad_avg', st, stream)

def save_adam(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('Adam', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(6,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        beta1, beta2 = pg['betas']
        floats[2] = beta1
        floats[3] = beta2
        floats[4] = pg['eps']
        floats[5] = pg['weight_decay']
        stream.write(floats.tobytes())
        _write_bool(pg['amsgrad'], stream)
        _write_bool(pg['maximize'], stream)

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            _write_tensor(st['exp_avg'], stream)
            _write_tensor(st['exp_avg_sq'], stream)
            _write_conditional_state_tensor('max_exp_avg_sq', st, stream)

def save_adamw(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('AdamW', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(6,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        beta1, beta2 = pg['betas']
        floats[2] = beta1
        floats[3] = beta2
        floats[4] = pg['eps']
        floats[5] = pg['weight_decay']
        stream.write(floats.tobytes())
        _write_bool(pg['amsgrad'], stream)
        _write_bool(pg['maximize'], stream)

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            _write_tensor(st['exp_avg'], stream)
            _write_tensor(st['exp_avg_sq'], stream)
            _write_conditional_state_tensor('max_exp_avg_sq', st, stream)

def save_nadam(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('NAdam', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(7,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        beta1, beta2 = pg['betas']
        floats[2] = beta1
        floats[3] = beta2
        floats[4] = pg['eps']
        floats[5] = pg['weight_decay']
        floats[6] = pg['momentum_decay']
        stream.write(floats.tobytes())

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            floats = np.empty(1,dtype=np.float64)
            floats[0] = float(st["mu_product"].item())       
            stream.write(floats.tobytes())
            _write_tensor(st['exp_avg'], stream)
            _write_tensor(st['exp_avg_sq'], stream)

def save_radam(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('RAdam', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(6,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        beta1, beta2 = pg['betas']
        floats[2] = beta1
        floats[3] = beta2
        floats[4] = pg['eps']
        floats[5] = pg['weight_decay']
        stream.write(floats.tobytes())

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            _write_tensor(st['exp_avg'], stream)
            _write_tensor(st['exp_avg_sq'], stream)

def save_adamax(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('Adamax', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(6,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        beta1, beta2 = pg['betas']
        floats[2] = beta1
        floats[3] = beta2
        floats[4] = pg['eps']
        floats[5] = pg['weight_decay']
        stream.write(floats.tobytes())

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            _write_tensor(st['exp_avg'], stream)
            _write_tensor(st['exp_inf'], stream)

def save_adadelta(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('Adadelta', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(5,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        floats[2] = pg['rho']
        floats[3] = pg['eps']
        floats[4] = pg['weight_decay']
        stream.write(floats.tobytes())
        _write_bool(pg['maximize'], stream)

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(st["step"], stream)
            _write_tensor(st['square_avg'], stream)
            _write_tensor(st['acc_delta'], stream)

def save_adagrad(optim, stream):

    sd = optim.state_dict()

    _write_optim_name('Adagrad', stream)

    _write_encoded_int64(len(optim.param_groups), stream)
    _write_encoded_int64(len(optim.state), stream)

    # Write options

    for pg in optim.param_groups:
        floats = np.empty(6,dtype=np.float64)
        floats[0] = pg['lr']        
        floats[1] = pg['lr']
        floats[2] = pg['lr_decay']
        floats[3] = pg['initial_accumulator_value']
        floats[4] = pg['eps']
        floats[5] = pg['weight_decay']
        stream.write(floats.tobytes())

    # Write state
    for group in optim.param_groups:
        for p in group['params']:
            st = optim.state[p]
            _write_int64(int(st["step"].item()), stream)
            _write_tensor(st['sum'], stream)
