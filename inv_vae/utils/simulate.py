import numpy as np
import networkx as nx 

def simulate_networks(n_nets=1000, n_nodes=68, p_in=.25, p_out=.01, seed=None,
                      add_noise=False, n_noisy_nets=100, noise_level=0.5, noise_std=0.01):
    '''
    simulate random community networks.
    ----
    inputs:
    
    n_nets: 
    n_nodes:
    p_in:
    p_out:
    add_noise:
    n_noisy_nets:
    noise_level:
    noise_std: 
    ----
    outputs: 
    
    sim_nets:
    noisy_nets:
    '''
    