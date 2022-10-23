# -------------------------------------------------------
# Encoderにlabel情報を与えるverのモデル定義（GPU用）
# -------------------------------------------------------
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim, coord_size):
        super(Encoder, self).__init__()  # オーバーライド用
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.coord_size = coord_size
        
        self.model = nn.Sequential(
            *block(1 + self.coord_size, 256, normalize=False),
            *block(256, 512),
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )
        self.l_mu = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.l_var = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, noise, labels):
        latent_input = torch.cat((labels, noise), -1)#label情報を付加
        latent = self.model(latent_input)
        latent = latent.view(latent.shape[0], self.latent_dim)
        mu = self.l_mu(latent)
        logvar = self.l_var(latent)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, coord_size):
        super(Decoder, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *block(latent_dim+1, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, coord_size),
            nn.Tanh()
        )

        self.coord_size = coord_size
        
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), -1)
        coords = self.model(gen_input)
        coords = coords.view(1, self.coord_size)
        return coords
        
class Discriminator(nn.Module):
    def __init__(self, coord_size):
        super(Discriminator, self).__init__()
        
        def block(in_feat, out_feat, dropout=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if dropout:
                layers.append(nn.Dropout(0.4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(1 + coord_size, 512, dropout=0.2),
            *block(512, 256, dropout=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, coords, labels):
        # Concatenate label embedding and image to produce input
        c_coords = torch.cat((coords.view(1, -1), labels), -1)
        c_coords_flat = c_coords.view(1, -1)
        validity = self.model(c_coords_flat)
        return validity