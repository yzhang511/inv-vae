import torch
from torch.autograd import Variable

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def to_var(x, requires_grad=False, volatile=False):
    """
    varialbe type that automatically choose cpu or cuda.
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile) 


def preprocess_neural_data(data):
    '''
    inputs:
    
    ----
    outputs:
    
    '''

    tensors = data['count_network']
    subject_traits = data['nih_cogn']
    subject_trans = data['all_motion'].reshape(-1,1)
    subject_rotat = data['all_rotation'].reshape(-1,1)
    subject_motion = np.vstack([subject_trans, subject_rotat])

    A_mat = np.mean(np.squeeze(tensor[18:86, 18:86,:,:]), axis=2)
    A_mat = A_mat + A_mat.transpose()

    offset = 100
    brain_networks = []
    for i in range(tensors.shape[-1]):
        network = np.float32(tensors[:,:,0,i] + np.transpose(tensors[:,:,0,i]))
        np.fill_diagonal(network, np.mean(network, 0))
        network = network[18:86, 18:86]
        network = network.flatten() / offset
        brain_networks.append(network)
    brain_networks = np.array(brain_networks)
    
    return brain_networks, subject_traits, subject_motion
