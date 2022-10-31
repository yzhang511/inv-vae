import numpy as np
import torch
from torch.autograd import Variable

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def min_max_standardize(x):
    return (x - x.min()) / (x.max() - x.min())

def to_var(x, requires_grad=False, volatile=False):
    """
    varialbe type that automatically choose cpu or cuda.
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile) 

def reset_weights(model):
    '''
    try resetting model weights to avoid weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def preprocess_neural_data(data_path):
    '''
    inputs:
    
    ----
    outputs:
    
    '''

    data = np.load(data_path, allow_pickle=True).item()
    count_networks = data['count_network']
    traits = data['trait']
    translations = data['translation']
    rotations = data['rotation']
    
    A_mat = np.mean(count_networks, axis=0)
    A_mat = A_mat + A_mat.transpose()
    
    offset = 100
    brain_networks = []
    for i in range(count_networks.shape[0]):
        network = np.float32(count_networks[i,:,:] + np.transpose(count_networks[i,:,:]))
        np.fill_diagonal(network, 0) # exclude self-connection from consideration
        network = network.flatten() / offset
        brain_networks.append(network)
    brain_networks = np.array(brain_networks)
    
    avg_trans = np.ma.masked_array(translations, mask=(translations == 0)).mean(axis=1).data.reshape(-1,1)
    avg_rotat = np.ma.masked_array(rotations, 
                                   mask=np.logical_or(rotations == 0, np.isnan(rotations))).mean(axis=1).data.reshape(-1,1)
    
    return brain_networks, traits, translations, rotations, avg_trans, avg_rotat
