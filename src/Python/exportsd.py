#
# Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#
import io
import torch
import leb128

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
        return 0
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
        write_tensor(sd[entry], stream)

