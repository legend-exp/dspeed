from __future__ import annotations

import numpy as np

from ..utils import GUFuncWrapper


def tf_model(filepath: str) -> GUFuncWrapper:
    """
    Initializer to load a tensorflow model to use in a ProcessingChain

    Parameters
    ----------
    filepath
        name of keras file containing model


    YAML Configuration Example
    --------------------------
    .. code-block:: yaml

        classifier:
          function: dspeed.processors.tf_model
          args:
            - input
            - output
          init_args:
            - "'model.keras'"
    """
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(filepath)

        # TODO: form the signature from the model inputs and outputs

        return GUFuncWrapper(
            lambda i: model(i).numpy().squeeze(),
            name=filepath,
            signature="(n)->()",
            types=["f->f", "d->d"],
            vectorized=True,
        )
    except OSError:
        return GUFuncWrapper(
            lambda i: np.nan,
            name="tfmodel_null",
            signature="(n)->()",
            types=["f->f", "d->d"],
        )
