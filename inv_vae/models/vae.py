import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from inv_vae.models.graph_conv import GraphConv
        

class VAE(nn.Module):
    '''
    graph auto-encoders. 
    ----
    inputs:
    
    ----
    outputs:
    
    '''
    def __init__(self, config): 
        super(VAE, self).__init__()
        self.n_nodes = config.n_nodes
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.n_enc_layers = config.n_enc_layers
        self.n_dec_layers = config.n_dec_layers
        self.drop_out = config.drop_out
        self.alpha=config.alpha
        self.beta=config.beta
        self.add_reg = config.add_reg
        self.y_dim = config.y_dim
        
        # encoder layers (inference model)
        self.W = Variable(torch.randn(self.n_dec_layers, 1), requires_grad=True)  # add cuda() if gpu available
        self.b = Variable(torch.randn(self.latent_dim * self.latent_dim), requires_grad=True) # add cuda() if gpu available
        enc_layers = [nn.Linear(self.latent_dim * self.latent_dim, self.hidden_dim) for i in range(self.n_enc_layers)]
        self.enc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_drop = nn.Dropout(p=self.drop_out) 
        self.encoder = nn.Sequential(*enc_layers)

        # decoder layers (generative model)        
        self.dec_layers = [nn.Linear(self.latent_dim, self.n_nodes) for i in range(self.n_dec_layers)]
        
        # graph convolution layers
        self.gc_layers = [GraphConv(self.n_nodes, self.n_nodes) for i in range(self.n_dec_layers)]
        
        self.fc = nn.Linear(self.n_nodes*self.n_nodes, self.n_nodes*self.n_nodes)
        
        if self.add_reg:
            self.reg = nn.Linear(self.latent_dim, self.y_dim)

    def encode(self, x_input):
        output = F.relu(self.encoder(x_input))
        output = self.enc_drop(output)
        return self.enc_mu(output), self.enc_logvar(output)

    def reparameterize(self, mu, logvar):
        sd = torch.exp(.5 * logvar)
        eps = torch.randn_like(sd)
        return mu + eps * sd

    def decode(self, z_input):
        dec_out = [torch.sigmoid(self.dec_layers[i](z_input)) for i in range(self.n_dec_layers)]    
        gc_out = [torch.sigmoid(self.gc_layers[i](dec_out[i])) for i in range(self.n_dec_layers)]     
        bmm_out = [torch.bmm(gc_out[i].unsqueeze(2), gc_out[i].unsqueeze(1)).view(-1, self.n_nodes*self.n_nodes, 1) \
                      for i in range(self.n_dec_layers)] 
        output = torch.cat(bmm_out, 2)
        output = torch.bmm(output, self.W.expand(output.shape[0], self.n_dec_layers, 1))
        output = output.view(-1, self.n_nodes*self.n_nodes) + self.b.expand(output.shape[0], self.n_nodes*self.n_nodes)
        output = torch.exp(self.fc(output))
        return output
            

    def forward(self, x_input):
        mu, logvar = self.encode(x_input.view(-1, self.n_nodes*self.n_nodes))
        z_sample = self.reparameterize(mu, logvar)
        x_output = self.decode(z_sample)
        if self.add_reg:
            y_output = self.reg(mu)
            return x_output, y_output, mu, logvar
        else:
            return x_output, mu, logvar

    def set_mask(self, masks):
        for i in range(self.n_dec_layers):
            self.gc_layers[i].set_mask(masks[i])
        
    def loss(self, x_output, x_input, mu, logvar):
        nll = F.poisson_nll_loss(x_output, x_input.view(-1, self.n_nodes*self.n_nodes), reduction='sum', log_input=False)
        kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = self.alpha * nll + self.beta * kl 
        return loss, nll, kl
    
    def reg_loss(self, x_output, x_input, y_output, y_input, mu, logvar):
        nll = F.poisson_nll_loss(x_output, x_input.view(-1, self.n_nodes*self.n_nodes), reduction='sum', log_input=False)
        kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mse = F.mse_loss(y_output, y_input, reduction='sum')
        loss = self.alpha * nll + self.beta * kl + mse
        return loss, nll, kl, mse
    
    def custom_train(self, epoch, train_loader, model, optimizer, device, n_epoch_display=5):
        model.train()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        n = len(train_loader.dataset)
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            x_input = batch_x.to(device)
            optimizer.zero_grad()
            x_output, mu, logvar = model(x_input)
            loss, nll, kl = model.loss(x_output, x_input, mu, logvar) 
            loss.backward()
            tot_loss += loss.item()
            tot_nll += nll.item()
            tot_kl += kl.item()
            optimizer.step()
        losses = [[tot_loss/n], [tot_nll/n], [tot_kl/n]]
        if (epoch % n_epoch_display) == 0:
            print('epoch: {} train loss: {:.3f} nll: {:.3f} kl: {:.3f}'.format(epoch, tot_loss/n, tot_nll/n, tot_kl/n))
        return losses
    
    def custom_test(self, epoch, test_loader, model, device, n_epoch_display=5):
        model.eval()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        n = len(test_loader.dataset)
        with torch.no_grad():
            for batch_idx, (batch_x, _) in enumerate(test_loader):
                x_input = batch_x.to(device)
                x_output, mu, logvar = model(x_input)
                loss, nll, kl = model.loss(x_output, x_input, mu, logvar) 
                tot_loss += loss.item()
                tot_nll += nll.item()
                tot_kl += kl.item()
            losses = [[tot_loss/n], [tot_nll/n], [tot_kl/n]]
            if (epoch % n_epoch_display) == 0:
                print('epoch: {} test loss {:.3f} nll: {:.3f} kl: {:.3f}'.format(epoch, tot_loss/n, tot_nll/n, tot_kl/n))
        return losses
    
    def reg_train(self, epoch, train_loader, model, optimizer, device, n_epoch_display=5):
        model.train()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        tot_mse = 0
        n = len(train_loader.dataset)
        for batch_idx, (batch_x, _, batch_y) in enumerate(train_loader):
            x_input = batch_x.to(device)
            y_input = batch_y.to(device)
            optimizer.zero_grad()
            x_output, y_output, mu, logvar = model(x_input)
            loss, nll, kl, mse = model.reg_loss(x_output, x_input, y_output, y_input, mu, logvar) 
            loss.backward()
            tot_loss += loss.item()
            tot_nll += nll.item()
            tot_kl += kl.item()
            tot_mse += mse.item()
            optimizer.step()
        losses = [[tot_loss/n], [tot_nll/n], [tot_kl/n], [tot_mse/n]]
        if (epoch % n_epoch_display) == 0:
            print('epoch: {} train loss: {:.3f} nll: {:.3f} kl: {:.3f} mse: {:.3f}'.format(
                epoch, tot_loss/n, tot_nll/n, tot_kl/n, tot_mse/n) )
        return losses
    
    def reg_test(self, epoch, test_loader, model, device, n_epoch_display=5):
        model.eval()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        tot_mse = 0
        n = len(test_loader.dataset)
        with torch.no_grad():
            for batch_idx, (batch_x, _, batch_y) in enumerate(test_loader):
                x_input = batch_x.to(device)
                y_input = batch_y.to(device)
                x_output, y_output, mu, logvar = model(x_input)
                loss, nll, kl, mse = model.reg_loss(x_output, x_input, y_output, y_input, mu, logvar) 
                tot_loss += loss.item()
                tot_nll += nll.item()
                tot_kl += kl.item()
                tot_mse += mse.item()
            losses = [[tot_loss/n], [tot_nll/n], [tot_kl/n], [tot_mse/n]]
            if (epoch % n_epoch_display) == 0:
                print('epoch: {} train loss: {:.3f} nll: {:.3f} kl: {:.3f} mse: {:.3f}'.format(
                    epoch, tot_loss/n, tot_nll/n, tot_kl/n, tot_mse/n) )
        return losses