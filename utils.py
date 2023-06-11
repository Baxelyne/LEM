# -*- coding: utf-8 -*-
from keras import backend as K
import numpy as np
from keras.utils import normalize
from keras import regularizers


def log(logfile, text, write_to_log=True):
    print(text)
    if write_to_log:
        with open(logfile, 'a') as f:
            f.write(text + '\n')


class CustomizedL1L2(regularizers.L1L2):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(K.cast_to_floatx(l1))
        self.l2 = K.variable(K.cast_to_floatx(l2))


def generate_arrays_from_source(sp_mat):
    arrays = np.array(list(map(lambda x: np.squeeze(np.asarray(x.todense())), sp_mat)))
    index_arrays = np.zeros_like(arrays, dtype="int32")
    index_arrays[arrays > 0] = 1
    return normalize(arrays), index_arrays




def print_weight_shape(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    for name, weight in zip(names, weights):
        print(name, weight.shape)

def check_sparsity(model, sparsity_threshold=1e-3):
    kernel = model.get_weights()[-2]
    num_weights = kernel.shape[0] * kernel.shape[1]
    num_zero = np.array(np.abs(kernel) < sparsity_threshold, dtype=float).sum()
    return num_zero / float(num_weights)

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    current_l1 = K.get_value(cur_l1.l1)
    diff = sparsity_target - cur_sparsity
    new_l1 = current_l1 * 2.0 ** diff
    K.set_value(cur_l1.l1, K.cast_to_floatx(new_l1))

