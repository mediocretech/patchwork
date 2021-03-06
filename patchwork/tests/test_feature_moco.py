# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._moco import copy_model, exponential_model_update
from patchwork.feature._moco import _build_augment_pair_dataset
from patchwork.feature._moco import _build_momentum_contrast_training_step
from patchwork.feature._moco import _build_logits

def test_copy_model():
    orig = tf.keras.applications.VGG16(weights=None, include_top=False)
    clone = copy_model(orig)
    # number of layers should be the same
    assert len(orig.layers) == len(clone.layers)
    # weights should be identical
    assert np.sum((orig.layers[1].get_weights()[0]-clone.layers[1].get_weights()[0])) == 0.


def test_exponenial_model_update():
    test_inpt = np.ones((1,5), dtype=np.float32)
    
    inpt = tf.keras.layers.Input((5))
    net = tf.keras.layers.Dense(4)(inpt)
    mod1 = tf.keras.Model(inpt, net)
    out1 = mod1(test_inpt).numpy()
    
    inpt = tf.keras.layers.Input((5))
    net = tf.keras.layers.Dense(4)(inpt)
    mod2 = tf.keras.Model(inpt, net)
    out2 = mod2(test_inpt).numpy()
    
    # updating model with itself should give rolling sum = 0
    rollingsum = exponential_model_update(mod1, mod1).numpy()
    assert rollingsum == 0.
    
    # setting exponential parameter to 1 should return original model
    rs = exponential_model_update(mod1, mod2, 1.)
    out3 = mod1(test_inpt).numpy()
    assert np.sum((out3 - out1)**2) < 1e-5
    
    # setting exponential parameter to 0 should return second model
    rs = exponential_model_update(mod1, mod2, 0.)
    out4 = mod1(test_inpt).numpy()
    assert np.sum((out4 - out2)**2) < 1e-5
    
    
def test_build_augment_pair_dataset(test_png_path):
    filepaths = 10*[test_png_path]
    ds = _build_augment_pair_dataset(filepaths, imshape=(32,32),
                                     batch_size=5, 
                                     augment={"flip_left_right":True})
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break
    
    assert x.shape == (5,32,32,3)
    assert y.shape == (5,32,32,3)
    
    
def test_build_augment_pair_dataset_with_custom_dataset():
    rawdata = np.zeros((10,32,32,3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(rawdata)
    batch_size = 5
    ds = _build_augment_pair_dataset(ds, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment={"flip_left_right":True}, 
                              single_channel=False,
                              batch_size=batch_size)
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break
    
    assert x.shape == (5,32,32,3)
    assert y.shape == (5,32,32,3)
    
    
def test_build_logits_no_mochi():
    batch_size = 7
    embed_dim = 5
    K = 13

    q = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    k = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    buffer = tf.Variable(tf.nn.l2_normalize(np.random.normal(0, 1, size=(K,embed_dim)).astype(np.float32), axis=1))
    all_logits = _build_logits(q, k, buffer, tf.GradientTape())
    assert len(all_logits.shape) == 2
    assert all_logits.shape[0] == batch_size
    assert all_logits.shape[1] == K +1
    
def test_build_logits_with_margin():
    batch_size = 7
    embed_dim = 5
    K = 13

    q = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    k = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    buffer = tf.Variable(tf.nn.l2_normalize(np.random.normal(0, 1, size=(K,embed_dim)).astype(np.float32), axis=1))
    all_logits = _build_logits(q, k, buffer, tf.GradientTape(), margin=100)
    assert len(all_logits.shape) == 2
    assert all_logits.shape[0] == batch_size
    assert all_logits.shape[1] == K +1
    
    
def test_build_logits_with_mochi():
    batch_size = 7
    embed_dim = 5
    K = 13
    N = 6
    s = 2

    q = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    k = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    buffer = tf.Variable(tf.nn.l2_normalize(np.random.normal(0, 1, size=(K,embed_dim)).astype(np.float32), axis=1))
    with tf.GradientTape() as tape:
        all_logits = _build_logits(q, k, buffer, tape, N, s)
    assert len(all_logits.shape) == 2
    assert all_logits.shape[0] == batch_size
    assert all_logits.shape[1] == K +1 + s
    
def test_build_logits_with_mochi_and_query_mixing():
    batch_size = 7
    embed_dim = 5
    K = 13
    N = 6
    s = 2
    s_prime = 3

    q = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    k = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    buffer = tf.Variable(tf.nn.l2_normalize(np.random.normal(0, 1, size=(K,embed_dim)).astype(np.float32), axis=1))
    with tf.GradientTape() as tape:
        all_logits = _build_logits(q, k, buffer, tape, N, s, s_prime)
    assert len(all_logits.shape) == 2
    assert all_logits.shape[0] == batch_size
    assert all_logits.shape[1] == K +1 + s + s_prime
    
    
def test_build_logits_with_mochi_and_more_samples_than_N():
    batch_size = 7
    embed_dim = 5
    K = 13
    N = 6
    s = 12
    s_prime = 13

    q = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    k = tf.nn.l2_normalize(np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32), axis=1)
    buffer = tf.Variable(tf.nn.l2_normalize(np.random.normal(0, 1, size=(K,embed_dim)).astype(np.float32), axis=1))
    with tf.GradientTape() as tape:
        all_logits = _build_logits(q, k, buffer, tape, N, s, s_prime)
    assert len(all_logits.shape) == 2
    assert all_logits.shape[0] == batch_size
    assert all_logits.shape[1] == K +1 + s + s_prime