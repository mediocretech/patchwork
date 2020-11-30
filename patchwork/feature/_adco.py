# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity
from patchwork.feature._moco import _build_augment_pair_dataset, _build_logits



def _build_adco_training_step(model, opt, buffer, tau=0.12,  weight_decay=0):
    """
    Function to build tf.function for a MoCo training step. Basically just follow
    Algorithm 1 in He et al's paper.
    """
    
    @tf.function
    def training_step(img1, img2):
        print("tracing training step")
        batch_size = img1.shape[0]
        #buffer_norm = tf.nn.l2_normalize(buffer, axis=1)
        outdict = {}
        with tf.GradientTape() as tape:
            # compute normalized embeddings for each 
            q = model(img1, training=True)
            k = model(img2, training=True)
            q = tf.nn.l2_normalize(q, axis=1)
            k = tf.nn.l2_normalize(k, axis=1)
            # compute logits
            all_logits = _build_logits(q, k, buffer, tape)
            # create labels (correct class is 0)- (N,)
            labels = tf.zeros((batch_size,), dtype=tf.int32)
            # compute crossentropy loss
            loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/tau))
                        
            if weight_decay > 0:
                l2_loss = compute_l2_loss(model)
                outdict["l2_loss"] = l2_loss
                loss += weight_decay*l2_loss
        # update model weights
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))
        # also compute the "accuracy"; what fraction of the batch has
        # the key as the largest logit. from figure 2b of the MoCHi paper
        nce_batch_accuracy = tf.reduce_mean(tf.cast(tf.argmax(all_logits, 
                                                              axis=1)==0, tf.float32))
        
        outdict["loss"] = loss
        outdict["nce_batch_accuracy"] = nce_batch_accuracy
        return outdict
    return training_step


def _build_adversarial_training_step(model, opt, buffer, tau=0.02):
    """
    Build tf.function for alternating batch
    """
    @tf.function
    def training_step(img1, img2):
        print("tracing adversarial training step")
        batch_size = img1.shape[0]
        # compute normalized embeddings for each 
        q = model(img1, training=True)
        k = model(img2, training=True)
        q = tf.nn.l2_normalize(q, axis=1)
        k = tf.nn.l2_normalize(k, axis=1)
        with tf.GradientTape() as tape:
            # compute logits
            all_logits = _build_logits(q, k, buffer, None)
            # create labels (correct class is 0)- (N,)
            labels = tf.zeros((batch_size,), dtype=tf.int32)
            # compute crossentropy loss
            loss =-1* tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/tau))
                        

        # update buffer weights
        gradients = tape.gradient(loss, [buffer])
        opt.apply_gradients(zip(gradients, [buffer]))
        # and normalize
        buffer.assign(tf.nn.l2_normalize(buffer, axis=1))
        
        return {"loss":loss}
    return training_step


class AdversarialContrastTrainer(GenericExtractor):
    """
    Class for training an Adversarial Contrast model.
    
    Based on "AdCo: Adversarial Contrast for Efficient Learning of
    Unsupervised Representations from Self-Trained Negative
    Adversaries" by Hu et al
    """
    modelname = "MomentumContrast"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, negative_examples=65536,
                 tau=0.12, adv_tau=0.02, output_dim=128, num_hidden=2048, 
                 weight_decay=1e-4,
                 lr=0.03, adv_lr=3.0, lr_decay=0, decay_type="exponential",
                 opt_type="momentum",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, strategy=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :negative_examples: number of adversarial negative examples
        :tau:.temperature parameter for noise-contrastive loss
        :adv_tau: temperature parameter for updating negative examples
        :batches_in_buffer:
        :num_hidden: number of neurons in the projection head's hidden layer (from the MoCoV2 paper)
        :weight_decay: L2 loss weight; 0 to disable
        :lr: (float) initial learning rate
        :adv_lr: (float) initial learning rate for adversarial updates
        :lr_decay: (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), or "cosine"
        :opt_type: (str) which optimizer to use; "momentum" or "adam"
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :num_parallel_calls: (int) number of threads for loader mapping
        :single_channel: if True, expect a single-channel input image and 
                stack it num_channels times.
        :notes: (string) any notes on the experiment that you want saved in the
                config.yml file
        :downstream_labels: dictionary mapping image file paths to labels
        :strategy:
        """
        assert augment is not False, "this method needs an augmentation scheme"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        self.strategy = strategy
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            # from "technical details" in paper- after FCN they did global pooling
            # and then a dense layer. i assume linear outputs on it.
            inpt = tf.keras.layers.Input((None, None, num_channels))
            features = fcn(inpt)
            pooled = tf.keras.layers.GlobalAvgPool2D()(features)
            # MoCoV2 paper adds a hidden layer
            dense = tf.keras.layers.Dense(num_hidden, activation="relu")(pooled)
            outpt = tf.keras.layers.Dense(output_dim)(dense)
            full_model = tf.keras.Model(inpt, outpt)
        

            self._models = {"fcn":fcn, "full":full_model}
        
        # build training dataset
        ds = _build_augment_pair_dataset(trainingdata, 
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            norm=norm, num_channels=num_channels, 
                            augment=augment, single_channel=single_channel)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizers for both steps
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        print("using vanilla SGD for negatives")
        self._adv_optimizer = self._build_optimizer(adv_lr, lr_decay,
                                                    opt_type="sgd",
                                                    decay_type=decay_type)
        
        # build buffer
        self._buffer = tf.Variable(np.zeros((negative_examples, output_dim), 
                                            dtype=np.float32))
        
        # build  and distribute both training steps
        step_fn = _build_adco_training_step(full_model, self._optimizer,
                                            self._buffer, tau=tau,
                                            weight_decay=weight_decay)
        adv_step_fn = _build_adversarial_training_step(full_model, 
                                                       self._adv_optimizer,
                                                       self._buffer, tau=adv_tau)        
        self._training_step = self._distribute_training_function(step_fn)
        self._adv_step = self._distribute_training_function(adv_step_fn)
        # build evaluation dataset
        if testdata is not None:
            self._test_ds = _build_augment_pair_dataset(testdata, 
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            norm=norm, num_channels=num_channels, 
                            augment=augment, single_channel=single_channel)
            self._test = True
        else:
            self._test = False

        self.step = 0
        
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, 
                            negative_examples=negative_examples,
                            tau=tau, adv_tau=adv_tau, 
                            output_dim=output_dim, num_hidden=num_hidden,
                            weight_decay=weight_decay,
                            lr=lr, adv_lr=adv_lr,  lr_decay=lr_decay,
                            opt_type=opt_type,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes,
                            trainer="adco")
        self._prepopulate_buffer()
        
    def _prepopulate_buffer(self):
        i = 0
        bs = self.input_config["batch_size"]
        K = self.config["negative_examples"]
        while i*bs < K:
            for x,y in self._ds:
                k = tf.nn.l2_normalize(
                        self._models["full"](y, training=True), axis=1)
                _ = self._buffer[bs*i:bs*(i+1),:].assign(k)
                i += 1
                if i*bs >= K:
                    break
        
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            if self.step % 2 == 0:
                losses = self._training_step(x,y)
                self._record_scalars(**losses)
                self._record_scalars(learning_rate=self._get_current_learning_rate())
            else:
                self._adv_step(x,y)
            
            self.step += 1
            
 
    def evaluate(self):
        if self._test:
            # if the user passed out-of-sample data to test- compute
            # alignment and uniformity measures
            alignment, uniformity = _compute_alignment_and_uniformity(
                                            self._test_ds, self._models["full"])
            
            self._record_scalars(alignment=alignment,
                             uniformity=uniformity, metric=True)
            metrics=["linear_classification_accuracy",
                                 "alignment",
                                 "uniformity"]
        else:
            metrics=["linear_classification_accuracy"]
            
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("tau", hp.RealInterval(0., 10000.)):self.config["tau"],
                    hp.HParam("adv_tau", hp.RealInterval(0., 10000.)):self.config["adv_tau"],
                    hp.HParam("negative_examples", hp.IntInterval(1, 1000000)):self.config["negative_examples"],
                    hp.HParam("output_dim", hp.IntInterval(1, 1000000)):self.config["output_dim"],
                    hp.HParam("num_hidden", hp.IntInterval(1, 1000000)):self.config["num_hidden"],
                    hp.HParam("weight_decay", hp.RealInterval(0., 10000.)):self.config["weight_decay"],
                    hp.HParam("lr", hp.RealInterval(0., 10000.)):self.config["lr"],
                    hp.HParam("adv_lr", hp.RealInterval(0., 10000.)):self.config["adv_lr"]
                    }
                for k in self.augment_config:
                    if isinstance(self.augment_config[k], float):
                        hparams[hp.HParam(k, hp.RealInterval(0., 10000.))] = self.augment_config[k]
            else:
                hparams=None
            self._linear_classification_test(hparams, metrics=metrics)
        
        
    def load_weights(self, logdir):
        """
        Update model weights from a previously trained model
        """
        super().load_weights(logdir)
        self._prepopulate_buffer()
            
        
