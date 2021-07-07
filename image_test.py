
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

def load_batch(batch_pkl):
    with open(batch_pkl, 'rb') as f:
        test_ds = pickle.load(f)
    return test_ds


def test_batch(): #test_ds has keys "image" and "label"
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test[:1%]', batch_size=-1, shuffle_files = True))
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    with open('test_batch.pkl', 'wb') as f:
        pickle.dump(test_ds, f)


def autoencoder_loss(logits, image):
   # if type(logits) or type(image) == NoneType:
    #    breakpoint()
    return jnp.mean((logits - image)**2)

def batch_loss(logits,images):
    loss_arr = (logits-images)**2
    loss = [jnp.mean(i) for i in loss_arr]
    return loss

def eval_image(opt_pkl,image):
    optimizer = load_optimizer(opt_pkl)
    logits = models.AutoEncoder().apply({'params': optimizer.target}, image)
    loss = autoencoder_loss(logits, image) 
    print(loss)

def eval_image_batch(opt_pkl,test_batch):
    eval_results = {'image': [],'label': [], 'logits': [], 'loss': [], 'latent': []}
    optimizer = load_optimizer(opt_pkl)
    model = models.AutoEncoder_sow()
    test_images = test_batch["image"]
    test_labels = test_batch["label"]

    #logits = models.AutoEncoder().apply({'params': optimizer.target}, test_images)
    logits, mod_vars = model.apply({'params': optimizer.target}, test_images, mutable=['intermediates'])
    loss = batch_loss(logits, test_images)
    latent = mod_vars["intermediates"]["latent"]

    eval_results['image'].append(test_images)
    eval_results['logits'].append(logits)
    eval_results['loss'].append(loss)
    eval_results['latent'].append(latent)
    eval_results['label'].append(test_labels)
    #print(loss)
    return eval_results

def eval_image_sow(opt_pkl,image):
    eval_results = {'image': [], 'logits': [], 'loss': [], 'latent': []}
    optimizer = load_optimizer(opt_pkl)
    model = models.AutoEncoder_sow()
    #logits = models.AutoEncoder_sow().apply({'params': optimizer.target}, image)
    logits, mod_vars = model.apply({'params': optimizer.target}, image, mutable=['intermediates'])
    loss = autoencoder_loss(logits, image)
    print(loss)
    print(mod_vars["intermediates"]["latent"])
    print(image)
    eval_results['image'].append(image)
    eval_results['logits'].append(logits)
    eval_results['loss'].append(loss)
    eval_results['latent'].append(mod_vars["intermediates"]["latent"])
    return eval_results

'''test_ds = load_batch()
eval_image_sow('opt_test.pkl', test_ds)'''

test_batch = load_batch('test_batch.pkl')
results = eval_image_batch('opt_test.pkl', test_batch)
for key,value in results.items():
    print(key, len(value))
