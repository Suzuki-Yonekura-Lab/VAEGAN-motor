if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt


def to_cuda(c):
  if torch.cuda.is_available():
    return c.cuda()

  return c

def to_cpu(c):
  if torch.cuda.is_available():
    return c.cpu()
  
  return c

def save_coords_motor(gen_coords, labels, path):
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = gen_coords[i][0]
        label = labels[i]
        # x = np.hstack([coord[:136], coord[272:272+184], coord[272+368:272+368+146]])
        # y = np.hstack([coord[136:272], coord[272+184:272+368], coord[272+368+146:]])
        x = np.hstack([coord[:136], coord[272:272+184]])
        y = np.hstack([coord[136:272], coord[272+184:272+368]])
        ax[i%4, i//4].plot(x,y)
        cl = round(label.item(), 4)
        title = 'CL={0}'.format(str(cl))
        ax[i%4, i//4].set_title(title)
    
    # plt.show()
    fig.savefig(path)
    plt.close()

def save_loss(Enc_losses, Dec_losses, Dis_losses, path="results/loss.png"):
    fig = plt.figure(figsize=(10,5))
    plt.title("Encoder, Decoder and Discriminator Loss During Training")
    plt.plot(Enc_losses,label="Enc")
    plt.plot(Dec_losses,label="Dec")
    plt.plot(Dis_losses,label="Dis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path)
    plt.close()
