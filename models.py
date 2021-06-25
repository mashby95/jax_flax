from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x

class AutoEncoder(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    # Encoder:
    x = nn.Conv(features=32, kernel_size=(3, 3))(x) #(1, 28, 28, 32)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3))(x) #(1, 28, 28, 64)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3))(x) #(1, 28, 28, 64)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten (1, 50176)
    x = nn.Dense(features = 256)(x) #(1, 256)
    x = nn.relu(x)
    x = nn.Dense(features = 10)(x) #(1, 10)
    encoded = nn.relu(x)


    #  Decoder:
    x = nn.Dense(features = 256)(encoded)
    x = nn.relu(x)
    x = nn.Dense(features = 50176)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], 28, 28, 64)) #hardcoded
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=1, kernel_size=(3, 3))(x)
    decoded = nn.sigmoid(x) #(1,28,28,1)
    return decoded

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x

class AutoEncoder_sow(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    # Encoder:
    x = nn.Conv(features=32, kernel_size=(3, 3))(x) #(1, 28, 28, 32)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3))(x) #(1, 28, 28, 64)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3))(x) #(1, 28, 28, 64)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten (1, 50176)
    x = nn.Dense(features = 256)(x) #(1, 256)
    x = nn.relu(x)
    x = nn.Dense(features = 10)(x) #(1, 10)
    x = nn.relu(x) #encoded
    self.sow('intermediates', 'latent',  x)


    #  Decoder:
    x = nn.Dense(features = 256)(x)
    x = nn.relu(x)
    x = nn.Dense(features = 50176)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], 28, 28, 64)) #hardcoded
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=1, kernel_size=(3, 3))(x)
    x = nn.sigmoid(x) #(1,28,28,1)
    return x #decoded

class Sow2(nn.Module):
    @nn.compact
    def __call__(self,x):
        mod = AutoEncoder_sow(name = 'AutoEncoder_sow')
        return mod(x) + mod(x)

def get_initial_params(key):
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  return initial_params


def create_optimizer(params, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(params)
  return optimizer
