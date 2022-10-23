import numpy as np
import networkx as nx

def simulate_networks(n_nets=1000, n_nodes=68, p_in=.25, p_out=.01, seed=None,
                      n_noisy_nets=100, noise_level=1.5, noise_sd=0.01):
    '''
    simulate noise-free and noise-affected random community networks.
    ----
    inputs:
    
    n_nets: total number of networks to be simulated.
    n_nodes: number of nodes in each simulated network.
    p_in: probability of edges with in groups; default to 2 groups.
    p_out: probability of edges between groups.
    n_noisy_nets: number of simulated networks that are significantly affected by noises.
    noise_level: amount of noises; noise ~ normal(noise_level, noise_sd). 
    noise_std: spread of noises. 
    ----
    outputs: 
    
    sim_nets: simulated noise-free networks; (n_nets, n_nodes, n_nodes) array.
    noisy_nets: noise-affected networks; (n_nets, n_nodes, n_nodes) array.
    noises: simulated noises. 
    noise_ids: indices for big and small noises.
    '''
    
    # simulate noises
    noises = np.zeros(n_nets)
    noise_ids = np.zeros_like(noises)
    noise_ids[:n_noisy_nets] = 1
    
    np.random.seed(seed)
    noises[:n_noisy_nets] = np.random.normal(noise_level, noise_sd, n_noisy_nets)    
    noises[n_noisy_nets:] = np.random.normal(1, noise_sd, n_nets-n_noisy_nets)
    
    # simulate noise-free and noise-affected networks
    sim_nets = []
    noisy_nets = []
    for i in range(n_nets):
        G = nx.random_partition_graph([n_nodes//2, n_nodes//2], p_in, p_out)
        A = nx.to_scipy_sparse_matrix(G).todense().reshape(n_nodes, n_nodes)
        noisy_A = np.matmul(noises[i]*A, noises[i]*A)
        noisy_A[range(n_nodes), range(n_nodes)] = 0
        sim_nets.append(A)
        noisy_nets.append(noisy_A)
    return np.array(sim_nets), np.array(noisy_nets), noises, noise_ids