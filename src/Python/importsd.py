#
# Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#
import torch
import leb128
import numpy as np
from collections import OrderedDict


_DTYPE_SIZE_MAP = {
    np.uint8: 1,
    np.int8: 1,
    np.int16: 2,
    np.int32: 4,
    np.int64: 8,
    np.float16: 2,
    np.float32: 4,
    np.float64: 8,
}


def _get_elem_type(type_num: int):
    if type_num == 0:
        return np.uint8
    elif type_num == 1:
        return np.int8
    elif type_num == 2:
        return np.int16
    elif type_num == 3:
        return np.int32
    elif type_num == 4:
        return np.int64
    elif type_num == 5:
        return np.float16
    elif type_num == 6:
        return np.float32
    elif type_num == 7:
        return np.float64
    elif type_num == 11:
        # return torch.bool
        raise NotImplemented("Unsupported data type")
    elif type_num == 15:
        # return torch.bfloat16
        raise NotImplemented("Unsupported data type")
    elif type_num == 4711:
        raise NotImplemented("Unsupported data type")
    else:
        raise ValueError("cannot decode the data type")


def load_state_dict(stream):
    """
    Loads a PyTorch state dictionary using the format that saved by TorchSharp.

    :param stream: An write stream opened for binary I/O.
    :return sd: A dictionary can be loaded by 'model.load_state_dict()'
    """
    sd = OrderedDict()
    dict_len, _ = leb128.u.decode_reader(stream)
    for i in range(dict_len):
        key_len, _ = leb128.u.decode_reader(stream)
        key_name = stream.read(key_len).decode("utf-8")

        ele_type, _ = leb128.u.decode_reader(stream)
        buffer_dtype = _get_elem_type(ele_type)

        buffer_shape_len, _ = leb128.u.decode_reader(stream)
        buffer_shape = tuple(leb128.u.decode_reader(stream)[0] for _ in range(buffer_shape_len))
        if buffer_shape:
            data_size = np.prod(buffer_shape)
        else:
            data_size = 1

        data_size_bytes = data_size * _DTYPE_SIZE_MAP[buffer_dtype]
        sd[key_name] = torch.from_numpy(
            np.frombuffer(
                stream.read(data_size_bytes), dtype=buffer_dtype, count=data_size
            ).reshape(buffer_shape)
        )
    return sd
