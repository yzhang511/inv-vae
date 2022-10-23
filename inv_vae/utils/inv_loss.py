import torch


def all_pairs_gaussian_kl(mu, sigma, eps=1e-8):
    '''
    
    '''
    sigma_sq = torch.square(sigma) + eps
    sigma_sq_inv = torch.reciprocal(sigma_sq)

    term1 = torch.mm(sigma_sq, torch.transpose(sigma_sq_inv, 0, 1))
    
    r = torch.mm(mu * mu, torch.transpose(sigma_sq_inv, 0, 1))
    r2 = mu * mu * sigma_sq_inv 
    r2 = torch.sum(r2, 1)

    term2 = 2 * torch.mm(mu, torch.transpose(mu*sigma_sq_inv, 0, 1))
    term2 = r - term2 + torch.transpose(r2.view(-1,1), 0, 1)
    
    r = torch.sum(torch.log(sigma_sq), 1)
    r = r.view(-1, 1)
    term3 = r - torch.transpose(r, 0, 1)
    
    return .5 * ( term1 + term2 + term3)

def kl_conditional_and_marg(mu, log_sigma_sq, latent_dim):
    '''
    
    '''
    sigma = torch.exp( .5 * log_sigma_sq )
    all_pairs_gkl = all_pairs_gaussian_kl(mu, sigma) - .5 * latent_dim
    
    return torch.mean(all_pairs_gkl)