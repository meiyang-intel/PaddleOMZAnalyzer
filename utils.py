import numpy as np

def str2bool(v):
    return v.lower() in ("true", "t", "1")

'''
paddle specific helpers
'''
import paddle

PADDLE_FLOAT_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64"
}
PADDLE_INT_DICT = {
    paddle.int8: "int8",
    paddle.int32: "int32",
    paddle.int64: "int64"    
}
PADDLE_DTYPE_DICT = {**PADDLE_FLOAT_DICT, **PADDLE_INT_DICT}

def is_float_tensor(dtype):
    """Is a float tensor"""
    return dtype in PADDLE_FLOAT_DICT.keys()

def get_tensor_dtype(dtype):
    assert dtype in PADDLE_DTYPE_DICT.keys()
    return PADDLE_DTYPE_DICT[dtype]

def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)    

