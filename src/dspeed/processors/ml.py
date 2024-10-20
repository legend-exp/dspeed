from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs

@guvectorize(
    ["void(float32[:], float32[:])", 
     "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def relu(x_in: np.ndarray, x_out: np.ndarray)-> None:
    """
    relu activation function 0 if x_in < 0 else x_in
    """
    x_out[:] =  x_in * (x_in > 0)

@guvectorize(
    ["void(float32[:], float32[:])", 
     "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def leaky_relu(x_in: np.ndarray, x_out: np.ndarray)-> None:
    """
    leaky relu activation function 0 if x_in < 0 else 0.01 x_in
    """
    x_out[:] =  x_in * (x_in > 0) + 0.01*x_in * (x_in < 0)


@guvectorize(
    ["void(float32[:], float32[:])", 
     "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def sigmoid(x_in: np.ndarray, x_out: np.ndarray)-> None:
    """
    sigmoid activation function
    """
    x_out[:] =  1/(1 + np.exp(-x_in))

@guvectorize(
    ["void(float32[:], float32[:])", 
     "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def softmax(x_in: np.ndarray, x_out: np.ndarray)-> None:
    """
    softmax activation function
    """
    x_out[:] =  np.log(1+np.exp(x_in))

@guvectorize(
    ["void(float32[::1], float32[:,::1], char, float32[:])", 
    "void(float64[::1], float64[:,::1], char, float64[:])"],
    "(n),(n,m),()->(m)",
    **nb_kwargs,
)
def dense_layer_no_bias(x_in: np.ndarray, 
                          kernel: np.ndarray, 
                          activation_func: np.int8, 
                          x_out: np.ndarray)-> None:

    """
    basic dense neural network layer with no bias 
    specify activation func using ord("s") where char can be:

    s - sigmoid
    r - relu
    l - leaky relu
    m - softmax
    t - tanh
    """

    x_out[:] = np.nan

    if np.isnan(x_in).any():
        return
    
    temp = np.dot(x_in, kernel)
    
    if activation_func == ord("s"):
        sigmoid(temp, x_out)
    elif activation_func == ord("r"):
        relu(temp, x_out)
    elif activation_func == ord("l"):
        leaky_relu(temp, x_out)
    elif activation_func == ord("m"):
        softmax(temp, x_out)
    elif activation_func == ord("t"):
        x_out[:] = np.tanh(temp)

@guvectorize(
    ["void(float32[::1], float32[:,::1], float32[:], char, float32[:])", 
    "void(float64[::1], float64[:,::1], float64[:], char, float64[:])"],
    "(n),(n,m),(m),()->(m)",
    **nb_kwargs,
)
def dense_layer_with_bias(x_in: np.ndarray, 
                          kernel: np.ndarray, 
                          bias: np.ndarray, 
                          activation_func: np.int8, 
                          x_out: np.ndarray)-> None:

    """
    basic dense neural network layer with bias added 
    specify activation func using ord("s") where char can be:

    s - sigmoid
    r - relu
    l - leaky relu
    m - softmax
    t - tanh
    """

    x_out[:] = np.nan

    if np.isnan(x_in).any():
        return
    
    temp = np.dot(x_in, kernel) + bias
    
    if activation_func == ord("s"):
        sigmoid(temp, x_out)
    elif activation_func == ord("r"):
        relu(temp, x_out)
    elif activation_func == ord("l"):
        leaky_relu(temp, x_out)
    elif activation_func == ord("m"):
        softmax(temp, x_out)
    elif activation_func == ord("t"):
        x_out[:] = np.tanh(temp)

@guvectorize(
    ["void(float32[::1], float32[::1], char, float32[:])", 
    "void(float64[::1], float64[::1], char, float64[:])"],
    "(n),(n),()->()",
    **nb_kwargs,
)
def classification_layer_no_bias(x_in: np.ndarray, 
                          kernel: np.ndarray, 
                          activation_func: np.int8, 
                          x_out: float)-> None:
    """
    this is the same as dense_layer_no_bias but the final output is a single number  
    specify activation func using ord("s") where char can be:

    s - sigmoid
    r - relu
    l - leaky relu
    m - softmax
    t - tanh
    """
    x_out[0] = np.nan

    if np.isnan(x_in).any():
        return

    temp = np.zeros(1, dtype=x_out.dtype)
    temp[0] = np.dot(x_in, kernel)

    if activation_func == ord("s"):
        sigmoid(temp, x_out)
    elif activation_func == ord("r"):
        relu(temp, x_out)
    elif activation_func == ord("l"):
        leaky_relu(temp, x_out)
    elif activation_func == ord("m"):
        softmax(temp, x_out)
    elif activation_func == ord("t"):
        x_out[0] = np.tanh(temp[0])

@guvectorize(
    ["void(float32[::1], float32[::1], float32, char, float32[:])", 
    "void(float64[::1], float64[::1], float64, char, float64[:])"],
    "(n),(n),(),()->()",
    **nb_kwargs,
)
def classification_layer_with_bias(x_in: np.ndarray, 
                          kernel: np.ndarray, 
                          bias: np.ndarray, 
                          activation_func: np.int8, 
                          x_out: float)-> None:
    """
    this is the same as dense_layer_with_bias but the final output is a single number  
    specify activation func using ord("s") where char can be:

    s - sigmoid
    r - relu
    l - leaky relu
    m - softmax
    t - tanh
    """
    x_out[0] = np.nan

    if np.isnan(x_in).any():
        return

    temp = np.zeros_like(x_out)
    temp[0] = np.dot(x_in, kernel) + bias

    if activation_func == ord("s"):
        sigmoid(temp, x_out)
    elif activation_func == ord("r"):
        relu(temp, x_out)
    elif activation_func == ord("l"):
        leaky_relu(temp, x_out)
    elif activation_func == ord("m"):
        softmax(temp, x_out)
    elif activation_func == ord("t"):
        x_out[0] = np.tanh(temp[0])

@guvectorize(
    ["void(float32[:], float32[:], float32[:], float32[:])", 
    "void(float64[:], float64[:], float64[:], float64[:])"],
    "(n),(n),(n)->(n)",
    **nb_kwargs,
)
def normalisation_layer(x_in: np.ndarray, 
                          means: np.ndarray, 
                          variances: np.ndarray, 
                          x_out: np.ndarray)-> None:
    """
    Normalisation layer
    """
    x_out[:] = (x_in - means) / np.sqrt(variances)