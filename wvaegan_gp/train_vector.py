# -------------------------------------------------------
# Encoderにlabel情報を与えるWVAEGAN-gp（GPU用）
# -------------------------------------------------------
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import time
import torch
from torch.autograd import Variable
from wvaegan_gp.model_vector import Encoder, Decoder, Discriminator
from util import save_loss, save_coords_motor

OUTPUT_DIR = 'wvaegan_gp/results'
DATA_PATH = '../motor_dataset_20221018st'

# ハイパーパラメータ
EPOCHS = 50000  # エポック数
BATCH_SIZE = 64  # バッチサイズ
LEARNING_RATE = 1e-5  # 学習率
LATENT_DIM = 1  # 潜在変数の数
N_CLASSES = 1  # クラスの数
COORD_SIZE = 932  # 座標の数

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=EPOCHS, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=LATENT_DIM, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=N_CLASSES, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=COORD_SIZE, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt = parser.parse_args()

coord_shape = (opt.channels, opt.coord_size)
cuda = torch.cuda.is_available()
lambda_gp = 10
rng = np.random.default_rng(0)
adversarial_loss = torch.nn.BCELoss()  # Loss functions

# Loss weight for gradient penalty
done_epoch = 0
encoder = Encoder(opt.latent_dim, opt.coord_size)
decoder = Decoder(opt.latent_dim, opt.coord_size)
discriminator = Discriminator(opt.coord_size)

if done_epoch > 0:
    E_PATH = f"{OUTPUT_DIR}/encoder_params_{done_epoch}"
    G_PATH = f"{OUTPUT_DIR}/decoder_params_{done_epoch}"
    D_PATH = f"{OUTPUT_DIR}/discriminator_params_{done_epoch}"

    encoder.load_state_dict(torch.load(E_PATH, map_location=torch.device('cpu')))
    encoder.eval()

    decoder.load_state_dict(torch.load(G_PATH, map_location=torch.device('cpu')))
    decoder.eval()

    discriminator.load_state_dict(torch.load(D_PATH, map_location=torch.device('cpu')))
    discriminator.eval()


if cuda:
    print("use GPU")
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()

# Configure data loader
torques_npz = np.load(f"{DATA_PATH}/labels.npz")
coords_npz = np.load(f"{DATA_PATH}/coords.npz")
torques = torques_npz[torques_npz.files[0]]
torque_mean = torques_npz[torques_npz.files[2]]
torque_std = torques_npz[torques_npz.files[3]]
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]

# perfs_npz = np.load("../motor_dataset_20221013st/labels.npz")
# coords_npz = np.load("../motor_dataset_20221013st/coords.npz")
# coords = coords_npz['coords']
# coord_mean = coords_npz['mean']
# coord_std = coords_npz['std']
# perfs = perfs_npz['torque']
# perf_mean = perfs_npz['mean']
# perf_std = perfs_npz['std']

max_torque = torques.max()
min_torque = torques.min()

dataset = torch.utils.data.TensorDataset(torch.tensor(coords), torch.tensor(torques))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_image(epoch=None, data_num=12):
    labels = rng.uniform(min_torque, max_torque, size=(data_num, opt.n_classes))
    labels = Variable(FloatTensor(labels))

    z = Variable(FloatTensor(rng.standard_normal(size=(data_num, coord_shape[1]))))
    #データセットをencode
    mus, log_variances = encoder(z, labels)
    variances = torch.exp(log_variances * 0.5)
    Z_p = Variable(FloatTensor(rng.standard_normal(size=(data_num, opt.latent_dim))))
    Z = Z_p * variances + mus

    en_coords = decoder(Z, labels).cpu().detach().numpy()

    if cuda:
        labels = labels.cpu()
    labels = labels.detach().numpy()
    # 標準化を戻す
    en_coords_destandardized = en_coords*coord_std+coord_mean
    labels_destandardized = labels * torque_std + torque_mean

    if epoch is not None:
        save_coords_motor(en_coords_destandardized, labels_destandardized, f"wvaegan_gp/coords/epoch_{str(epoch).zfill(3)}_dim{opt.latent_dim}_vector")
    else:
        np.savez(f"{OUTPUT_DIR}/final_dim{opt.latent_dim}_vector", labels_destandardized, en_coords_destandardized)
        save_coords_motor(en_coords_destandardized, labels_destandardized, f"wvaegan_gp/coords/final_dim{opt.latent_dim}_vector.png")

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for VAEGAN"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(rng.random(size=(real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Training
# ----------
start = time.time()
enc_losses, dec_losses, dis_losses = [], [], []
batches_done = 0
for epoch in range(opt.n_epochs-done_epoch):
    epoch+=done_epoch
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=True)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=True)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_dis.zero_grad()
        
        # Configure input
        real_imgs = Variable(coords.type(FloatTensor))
        labels = Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes)))
        if cuda:
            labels = labels.cuda()

        #ランダムノイズから生成
        Z_p = Variable(FloatTensor(rng.standard_normal((batch_size, opt.latent_dim))))
        X_gen = decoder(Z_p, labels)
        
        # Loss for real images
        validity_disc_real = discriminator(real_imgs, labels)
        # Loss for fake images
        validity_disc_fake = discriminator(X_gen, labels)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.reshape(real_imgs.shape[0], *coord_shape).data, X_gen.data, labels)
        dis_loss = -torch.mean(validity_disc_real) + torch.mean(validity_disc_fake) + lambda_gp * gradient_penalty
        dis_loss.backward()
        optimizer_dis.step()
        
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        
        # -----------------------------------------------------
        # Train encoder and decoder every n_critic iterations
        # -----------------------------------------------------
        if i % opt.n_critic == 0:
            # ---------------------------
            #  Train Encoder and Decoder
            # --------------------------
            #データセットをencode
            mus, log_variances = encoder(real_imgs, labels)
            variances = torch.exp(log_variances * 0.5)
            Z_p = Variable(FloatTensor(rng.standard_normal(size=(batch_size, opt.latent_dim))))
            Z = Z_p * variances + mus
            
            #再構成データ
            X_recon = decoder(Z, labels)
            
            kl_div = -0.5*torch.sum(-log_variances.exp() - torch.pow(mus, 2) + log_variances +1)
            latent_loss = torch.sum(kl_div).clone().detach().requires_grad_(True)
            validity_real = discriminator(real_imgs, labels) # Loss for real images
            validity_fake = discriminator(X_recon, labels) # Loss for fake images
            discrim_layer_recon_loss = torch.mean(torch.square(validity_real - validity_fake)).clone().detach().requires_grad_(True)
            
            
            # Sample noise and labels as decoder input
            Z_p = Variable(FloatTensor(rng.standard_normal(size=(batch_size, opt.latent_dim))))
            labels_p = Variable(FloatTensor(rng.uniform(min_torque, max_torque, size=(batch_size, opt.n_classes))))
            X_p = decoder(Z_p, labels_p)
        
            validity_gen_fake = discriminator(X_p, labels_p)
            
            # train encoder
            enc_loss = latent_loss + discrim_layer_recon_loss
            enc_loss.backward(retain_graph=True)
            optimizer_enc.step()
            
            # train decoder
            dec_loss = -torch.mean(validity_gen_fake) + discrim_layer_recon_loss
            dec_loss.backward(retain_graph=True)
            optimizer_dec.step()
        
        
            if i==0:
                print(
                    "[Epoch %d/%d %ds] [Enc loss: %f] [Dec loss: %f] [Dis loss: %f]"
                    % (epoch+1, opt.n_epochs,  int(time.time()-start), enc_loss.item(), dec_loss.item(), dis_loss.item())
                )
        
                enc_losses.append(enc_loss.item())
                dec_losses.append(dec_loss.item())
                dis_losses.append(dis_loss.item())
                
            batches_done += opt.n_critic
    
    if epoch % 5000 == 0:
        torch.save(encoder.state_dict(), f"{OUTPUT_DIR}/encoder_params_dim{opt.latent_dim}_vector_{epoch}")
        torch.save(decoder.state_dict(), f"{OUTPUT_DIR}/decoder_params_dim{opt.latent_dim}_vector_{epoch}")
        torch.save(discriminator.state_dict(), f"{OUTPUT_DIR}/discriminator_params_dim{opt.latent_dim}_vector_{epoch}")
    if epoch % 5000 == 0:
        sample_image(epoch=epoch)

        

torch.save(encoder.state_dict(), f"{OUTPUT_DIR}/encoder_params_dim{opt.latent_dim}_vector_{opt.n_epochs+done_epoch}")
torch.save(decoder.state_dict(), f"{OUTPUT_DIR}/decoder_params_dim{opt.latent_dim}_vector_{opt.n_epochs+done_epoch}")
torch.save(discriminator.state_dict(), f"{OUTPUT_DIR}/discriminator_params_dim{opt.latent_dim}_vector_{opt.n_epochs+done_epoch}")
sample_image(data_num=100)
save_loss(enc_losses, dec_losses, dis_losses, path=f"{OUTPUT_DIR}/loss_dim{opt.latent_dim}_vector.png")