
from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow_datasets as tfds
import pickle
from flax import serialization
import models
import random


def load_optimizer(opt_pkl):
    with open(opt_pkl, 'rb') as f:
        opt_dict = pickle.load(f)
    optimizer = models.create_optimizer(opt_dict['target'], 0.01, 0.9)
    optimizer = serialization.from_state_dict(optimizer, opt_dict)
    return optimizer

def load_image():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1, shuffle_files = True))
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    image = random.choice(test_ds['image'])
    image = image.reshape(1,28,28,1)
    print(f'Image dimensions = {image.shape}')
    return image #(1,28,28,1)

def autoencoder_loss(logits, image):
   # if type(logits) or type(image) == NoneType:
    #    breakpoint()
    return jnp.mean((logits - image)**2)

def eval_image(opt_pkl,image):
    optimizer = load_optimizer(opt_pkl)
    logits = models.AutoEncoder().apply({'params': optimizer.target}, image)
    loss = autoencoder_loss(logits, image) 
    print(loss)


image = load_image()
eval_image('opt_test.pkl', image)
