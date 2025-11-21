"""
Module containing ml processors, the dsp config can be used to combine these into a neural
network a simple example would be:

.. code-block:: yaml

    layer_1:
      function: normalisation_layer
      module: dspeed.processors
      args:
        - wf_blsub
        - db.mean
        - db.variance
        - layer_1
    layer_2:
      function: dense_layer_with_bias
      module: dspeed.processors
      args:
        - layer_1
        - db.kernel
        - db.bias
        - "'r'"
        - layer_2
    classifier:
      function: dense_layer_with_bias
      module: dspeed.processors
      args:
        - layer_2
        - db.kernel
        - db.bias
        - "'s'"
        - classifier
"""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def relu(x_in: np.ndarray, x_out: np.ndarray) -> None:
    """
    relu activation function 0 if x_in < 0 else x_in
    """
    x_out[:] = x_in * (x_in > 0)


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def leaky_relu(x_in: np.ndarray, x_out: np.ndarray) -> None:
    """
    leaky relu activation function 0 if x_in < 0 else 0.01 x_in
    """
    x_out[:] = x_in * (x_in > 0) + 0.01 * x_in * (x_in < 0)


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def sigmoid(x_in: np.ndarray, x_out: np.ndarray) -> None:
    """
    sigmoid activation function
    """
    x_out[:] = 1 / (1 + np.exp(-x_in))


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def softmax(x_in: np.ndarray, x_out: np.ndarray) -> None:
    """
    softmax activation function
    """
    x_out[:] = np.log(1 + np.exp(x_in))


@guvectorize(
    [
        "void(float32[::1], float32[:,::1], char, float32[:])",
        "void(float64[::1], float64[:,::1], char, float64[:])",
    ],
    "(n),(n,m),()->(m)",
    **nb_kwargs,
    forceobj=True,
)
def dense_layer_no_bias(
    x_in: np.ndarray, kernel: np.ndarray, activation_func: np.int8, x_out: np.ndarray
) -> None:
    """
    Basic dense neural network layer with no bias, f(x.W)

    Parameters
    ----------
    w_in
        the input waveform shape n.
    kernel
        the matrix of weights shape (n x m).
    activation_func
        the activation function to use specify with char:
        s - sigmoid
        r - relu
        l - leaky relu
        m - softmax
        t - tanh
    x_out
        the output vector shape m.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        layer_1:
          function: dense_layer_no_bias
          module: dspeed.processors
          args:
            - wf_blsub
            - db.kernel
            - "'s'"
            - layer_1
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
    [
        "void(float32[::1], float32[:,::1], float32[:], char, float32[:])",
        "void(float64[::1], float64[:,::1], float64[:], char, float64[:])",
    ],
    "(n),(n,m),(m),()->(m)",
    **nb_kwargs,
    forceobj=True,
)
def dense_layer_with_bias(
    x_in: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    activation_func: np.int8,
    x_out: np.ndarray,
) -> None:
    """
    Basic dense neural network layer with bias added f(x.W+b)

    Parameters
    ----------
    w_in
        the input waveform shape n.
    kernel
        the matrix of weights shape (n x m).
    bias
        the bias with shape m.
    activation_func
        the activation function to use specify with char:
        s - sigmoid
        r - relu
        l - leaky relu
        m - softmax
        t - tanh
    x_out
        the output vector shape m.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        layer_1:
          function: dense_layer_with_bias
          module: dspeed.processors
          args:
            - wf_blsub
            - db.kernel
            - db.bias
            - "'s'"
            - layer_1
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
    [
        "void(float32[::1], float32[::1], char, float32[:])",
        "void(float64[::1], float64[::1], char, float64[:])",
    ],
    "(n),(n),()->()",
    **nb_kwargs,
    forceobj=True,
)
def classification_layer_no_bias(
    x_in: np.ndarray, kernel: np.ndarray, activation_func: np.int8, x_out: float
) -> None:
    """
    This is the same as dense_layer_no_bias but the final output is a single number  f(x.W)

    Parameters
    ----------
    w_in
        the input waveform shape n.
    kernel
        the matrix of weights shape (n x 1).
    activation_func
        the activation function to use specify with char:
        s - sigmoid
        r - relu
        l - leaky relu
        m - softmax
        t - tanh
    x_out
        the output value.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        classifier:
          function: dense_layer_with_bias
          module: dspeed.processors
          args:
            - wf_blsub
            - db.kernel
            - "'s'"
            - classifier
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
    [
        "void(float32[::1], float32[::1], float32, char, float32[:])",
        "void(float64[::1], float64[::1], float64, char, float64[:])",
    ],
    "(n),(n),(),()->()",
    **nb_kwargs,
    forceobj=True,
)
def classification_layer_with_bias(
    x_in: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    activation_func: np.int8,
    x_out: float,
) -> None:
    """
    this is the same as dense_layer_with_bias but the final output is a single number f(x.W+bs)

    Parameters
    ----------
    w_in
        the input waveform shape n.
    kernel
        the matrix of weights shape (n x 1).
    bias
        the bias in this case a single value.
    activation_func
        the activation function to use specify with char:
        s - sigmoid
        r - relu
        l - leaky relu
        m - softmax
        t - tanh
    x_out
        the output value.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        classifier:
          function: dense_layer_with_bias
          module: dspeed.processors
          args:
            - wf_blsub
            - db.kernel
            - db.bias
            - "'s'"
            - classifier
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
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(n),(n)->(n)",
    **nb_kwargs,
    forceobj=True,
)
def normalisation_layer(
    x_in: np.ndarray, means: np.ndarray, variances: np.ndarray, x_out: np.ndarray
) -> None:
    """
    Normalisation layer, (x_in - mu)/np.sqrt(variance)
    Note this is variance not standard deviation

    Parameters
    ----------
    w_in
        the input waveform shape n.
    means
        array of means for each input value shape n.
    variances
        array of variances for each input value shape n.
    x_out
        the output vector shape n.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_normed:
          function: normalisation_layer
          module: dspeed.processors
          args:
            - wf_blsub
            - db.mean
            - db.variance
            - wf_normed
    """
    x_out[:] = (x_in - means) / np.sqrt(variances)
