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
