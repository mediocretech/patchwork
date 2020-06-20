# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from patchwork.feature._byol import _build_byol_dataset
from patchwork.feature._byol import _build_byol_training_step
from patchwork.feature._byol import _build_models

# build a tiny FCN for testing
inpt = tf.keras.layers.Input((None, None, 3))
net = tf.keras.layers.Conv2D(5,1)(inpt)
net = tf.keras.layers.MaxPool2D(10,10)(net)
fcn = tf.keras.Model(inpt, net)



def test_byol_dataset(test_png_path):
    filepaths = 10*[test_png_path]
    batch_size = 5
    ds = _build_byol_dataset(filepaths, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment=True, single_channel=False,
                              batch_size=batch_size)
    
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        break
    # I don't actually do anything with y- just keeping a common
    # output structure across all the feature extractor datasets.
    # x should be a tuple of two different "views" (augmentations)
    # of a single image
    assert len(x) == 2
    assert x[0].shape == (batch_size, 32, 32, 3)
    


def test_build_models():
    models = _build_models(fcn, (32,32), 3, 17, 11)
    assert len(models) == 4
    for m in ["fcn", "online", "prediction", "target"]:
        assert isinstance(models[m], tf.keras.Model)
    
    assert models["prediction"].output_shape[-1] == 11
    assert models["target"].output_shape[-1] == 11
    
    
    
def test_build_byol_training_step():
    models = _build_models(fcn, (32,32), 3, 17, 11)
    opt = tf.keras.optimizers.SGD()
    step = _build_byol_training_step(models["online"], 
                                       models["prediction"],
                                       models["target"], 
                                       opt, 0.996, 
                                       weight_decay=1e-6)
    
    x1 = tf.zeros((4,32,32,3), dtype=tf.float32)
    x = (x1,x1)
    y = np.array([1,1,1,1]).astype(np.int32)
    lossdict = step(x,y)
    
    assert isinstance(lossdict["loss"].numpy(), np.float32)
    # should include total loss, the MSE component, and the L2 loss
    assert len(lossdict) == 3