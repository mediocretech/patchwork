# -*- coding: utf-8 -*-
"""

                stateless.py
                
                
Stateless training and active learning wrapper functions
for integrating with an app.


"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.stats as st

from patchwork._sample import stratified_sample, find_labeled_indices
from patchwork._sample import PROTECTED_COLUMN_NAMES
from patchwork._badge import KPlusPlusSampler, _build_output_gradient_function
from patchwork._losses import masked_binary_crossentropy
from patchwork._util import shannon_entropy

import sklearn.preprocessing, sklearn.decomposition, sklearn.cluster


def _build_model(input_dim, num_classes, num_hidden_layers=0, 
                 hidden_dimension=128,
                 normalize_inputs=False, dropout=0):
    """
    Macro to generate a Keras classification model
    """
    inpt = tf.keras.layers.Input((input_dim))
    net = inpt
    
    # if we're normalizing inputs:
    if normalize_inputs:
        norm = tf.keras.layers.Lambda(lambda x:K.l2_normalize(x,axis=1))
        net = norm(net)
        
    # for each hidden layer
    for _ in range(num_hidden_layers):
        if dropout > 0:
            net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(hidden_dimension, activation="relu")(net)
        
    # final layer
    if dropout > 0:
        net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(num_classes, activation="relu")(net)
    
    return tf.keras.Model(inpt, net)

def _build_training_dataset(features, df, num_classes, num_samples, 
                            batch_size):
    indices, labels = stratified_sample(df, num_samples, 
                                            return_indices=True)
    ds = tf.data.Dataset.from_tensor_slices((features[indices], labels))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds



def _labels_to_dataframe(labels, classes=None):
    """
    Input labels as a list of dictionaries; return same information as
    a pandas dataframe.
    
    :classes: if a list of strings is passed here, will prune the columns
        to just these categories
    """
    # convert to dataframe
    df = pd.DataFrame(labels)
    
    # if exclude and validation not present, add them
    for c in ["exclude", "validation"]:
        if c not in df.columns:
            df[c] = False
            
    if "filepath" not in df.columns:
        df["filepath"] = ""
            
    # prune if necessary
    if classes is not None:
        for c in df.columns:
            if c not in classes+PROTECTED_COLUMN_NAMES:
                df = df.drop(c, axis=1)
    # make sure columns are in the same order
    if classes is not None:
        df = df.reindex(columns=["filepath", "exclude", 
                                 "validation"]+classes)     
    return df


def _estimate_accuracy(tp, fp, tn, fn, alpha=5, beta=5):
    """
    Compute a set of estimates around the accuracy, using a
    beta-binomial model.
    
    :tp: count of true positives
    :fp: count of false positives
    :tn: count of true negatives
    :fn: count of false negatives
    :alpha, beta: parameters of Beta prior
    
    Returns dictionary containing
    :accuracy: point estimate of accuracy
    :interval_low: low end of 90% credible interval of accuracy
    :interval_high: high end of 90% credible interval of accuracy
    :base_rate: frequentist point estimate of the base rate
    :prob_higher_than_base_rate: estimate of the probability that
        accuracy is higher than the base rate
    """
    # total number
    N = tp + tn + fp + fn
    # accuracy point estimate
    #acc = (tp+tn)/N
    acc = (alpha+tp+tn)/(alpha + beta + tp + tn + fp + fn)
    # how many right/wrong
    num_right = tp+tn
    num_wrong = fp+fn
    # estimate base rate
    frac_pos = (tp+fn)/N
    base_rate = max(frac_pos, 1-frac_pos)
    
    interval_low = st.beta.ppf(0.05, alpha+num_right, beta+num_wrong)
    interval_high = st.beta.ppf(0.95, alpha+num_right, beta+num_wrong)
    prob_above_base_rate = 1-st.beta.cdf(base_rate, alpha+num_right, 
                                         beta+num_wrong)
    
    return {"accuracy":acc, "base_rate":base_rate,
           "interval_low":interval_low, "interval_high":interval_high,
           "prob_above_base_rate":prob_above_base_rate}


def _eval(features, df, classes, model, threshold=0.5, alpha=5,beta=5):
    """
    Macro for evaluating a classification model. does the following:
        
        1) find the subset of tags marked "validation"
        2) generate model predictions on the validation features
        3) round predictions to 1 or 0
        4) for each category:
            a) check to see whether the labels include at least one positive
                and one negative example
            b) if so, call _estimate_accuracy() to get performance measures
        5) return everything as a dict
    """
    outdict = {}
    # boolean array
    val_subset = (df["validation"] == True).values
    # if there aren't any validation points, return empty dictionaries
    if val_subset.sum() == 0:
        return {c:{} for c in classes}
    # (num_val, num_classes) array
    labels = df[classes].values[val_subset,:]
    # get model sigmoid predictions
    predictions = model.predict(features[val_subset,:])
    # round to 1 or 0
    predictions = (predictions >= threshold).astype(int)
    
    # for each class
    for i,c in enumerate(classes):
        # only bother computing if we have at least one positive
        # and one negative val example in this category
        if (1 in labels[:,i])&(0 in labels[:,i]):
            # true positives
            tp = np.sum((labels[:,i]==1)&(predictions[:,i] == 1))
            # true negatives
            tn = np.sum((labels[:,i]==0)&(predictions[:,i] == 0))
            # false positives
            fp = np.sum((labels[:,i]==0)&(predictions[:,i] == 1))
            # false negatives
            fn = np.sum((labels[:,i]==1)&(predictions[:,i] == 0))
            outdict[c] = _estimate_accuracy(tp,fp,tn,fn, alpha, beta)
        else:
            outdict[c] = {}
            
    return outdict
        

def train(features, labels, classes, training_steps=1000, 
          batch_size=32, learning_rate=1e-3, **model_kwargs):
    """
    Hey now, you're an all-star. Get your train on.
    
    
    :features: (num_samples, input_dim) array of feature vectors
    :training_steps: how many steps to train for
    :classes: list of strings; names of categories to include in model
    :batch_size: number of examples per batch
    :learning_rate: learning rate for Adam optimizer
    :model_kwargs: keyword arguments for model construction
    
    Returns
    :model: trained Keras model
    :training_loss: array of length (training_steps) giving the
        loss function value at each training step
    :validation_metrics: dictionary of validation metrics NOT YET
        IMPLEMENTED
    """
    # convert labels to a dataframe
    df = _labels_to_dataframe(labels, classes)
    
    num_classes = len(classes)
    input_dim = features.shape[1]
    
    # create a model and optimizer
    model = _build_model(input_dim, num_classes, **model_kwargs)
    opt = tf.keras.optimizers.Adam(learning_rate)
    
    # build a dataset for training
    num_samples = training_steps*batch_size
    ds = _build_training_dataset(features, df, num_classes, num_samples, 
                                 batch_size)
        
    # train the model, recording loss at each step
    training_loss = []
    @tf.function
    def training_step(x,y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = masked_binary_crossentropy(y, pred)
        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))
        return loss
        
    for x, y in ds:
        training_loss.append(training_step(x,y).numpy())
        
    # finally run some performance metrics
    acc = _eval(features, df, classes, model)
        
    return model, np.array(training_loss), acc



def get_indices_of_tiles_in_predicted_class(features, model, category_index, 
                                            threshold=0.5):
    """
    :features: (num_samples, input_dim) array of feature vectors
    :model: trained Keras classifier model
    :category_index: integer index of category to query
    :threshold: float between 0 and 1; minimum probability assessed by
        classifier
    """
    predictions = model.predict(features)
    category_predictions = predictions[:,category_index]
    return np.arange(features.shape[0])[category_predictions >= threshold]




def sample_random(labels, max_to_return=None):
    """
    Generate a random sample of indices
    
    :labels: list of dictionaries containing labels
    :max_to_return: if not None; max number of indices to return
    """
    N = len(labels)
    if max_to_return is None:
        max_to_return = N
    # create a list of unlabeled indices    
    df = _labels_to_dataframe(labels)
    labeled = list(find_labeled_indices(df))
    indices_to_sample_from = [n for n in range(N) if n not in labeled]
    # update num_to_return in case not many unlabeled are left
    max_to_return = min(max_to_return, len(indices_to_sample_from))
    
    return np.random.choice(indices_to_sample_from,size=max_to_return,
                            replace=False)

def sample_uncertainty(labels, features, model, max_to_return=None):
    """
    Return indices sorted by decreasing entropy.
    
    :features: (num_samples, input_dim) array of feature vectors
    :model: trained Keras classifier model
    :max_to_return: if not None; max number of indices to return
    """  
    N = features.shape[0]
    # get model predictions
    predictions = model.predict(features)
    # compute entropies for each prediction
    entropy = shannon_entropy(predictions)
    
    # create a list of unlabeled indices    
    df = _labels_to_dataframe(labels)
    labeled = list(find_labeled_indices(df))
    
    # order by decreasing entropy
    ordering = entropy.argsort()[::-1]
    ordered_indices = np.arange(N)[ordering]
    # prune out labeled indices
    ordered_indices = [i for i in ordered_indices if i not in labeled]
    # and clip list if necessary
    if max_to_return is not None:
        ordered_indices = ordered_indices[:max_to_return]
    return ordered_indices

def sample_diversity(labels, features, max_to_return=20):
    """
    Attempt to create a diverse, model-independent sample of
    images to tag. PCA-reduce feature vectors an then use k-means
    for sampling.
    
    :features: (num_samples, input_dim) array of feature vectors
    :max_to_return: if not None; max number of indices to return
    """  
    N = features.shape[0]

    # figure out which indices are yet labeled
    df = _labels_to_dataframe(labels)
    labeled = find_labeled_indices(df)
    
    # rescale, PCA reduce, and cluster feature vectors
    scaled = sklearn.preprocessing.StandardScaler().fit_transform(features)
    reduced = sklearn.decomposition.PCA(16).fit_transform(scaled)
    kmeans = sklearn.cluster.KMeans(n_clusters=max_to_return).fit(reduced)

    # now use this for sample- pull up to one not-yet-labeled
    # data point from each cluster.
    all_indices = np.arange(N)
    indices = []
    for i in range(max_to_return):
        subset = [j for j in all_indices[kmeans.labels_ == i]
                  if j not in labeled]
        if len(subset) > 0:
            indices.append(np.random.choice(subset))
    return np.array(indices)



def sample_badge(labels, features, model, max_to_return=None):
    """
    Use a trained model to compute output-gradient vectors for the
    BADGE algorithm for active learning. Return sorted indices.
    
    :features: (num_samples, input_dim) array of feature vectors
    :model: trained Keras classifier model
    :max_to_return: if not None; max number of indices to return
    """  
    N = features.shape[0]
    if max_to_return is None:
        max_to_return = N
        
    # compute badge embeddings- define a tf.function for it
    compute_output_gradients = _build_output_gradient_function(model)
    # then run that function across all the images.
    output_gradients = tf.map_fn(compute_output_gradients, features).numpy()
    
    # figure out which indices are yet labeled
    df = _labels_to_dataframe(labels)
    labeled = list(find_labeled_indices(df))
    # update max_to_return in case there aren't very many 
    # unlabeled indices left
    max_to_return = min(max_to_return, N-len(labeled))

    # initialize a K++ sampler
    badge_sampler = KPlusPlusSampler(output_gradients, indices=labeled)
    return badge_sampler(max_to_return)






