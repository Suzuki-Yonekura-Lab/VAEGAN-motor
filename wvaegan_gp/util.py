if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

def save_coords_motor(gen_coords, labels, path):
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), figsize=(12, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.4)
    torque_idx = np.argsort(labels.flatten())
    for i in range(min(20, data_size)):
        idx = torque_idx[i]

        coord = gen_coords[idx][0]
        label = labels[idx][0]
        
        ax[i%4, i//4].plot(coord[:136], coord[136:272])  # 磁石
        ax[i%4, i//4].plot(coord[272:272+184], coord[272+184:272+368])  # 穴
        # ax[i%4, i//4].plot(coord[272+368:272+368+146], coord[272+368+146:])  # 外枠

        torque = round(label.item(), 3)
        title = f'torque={torque}'
        ax[i%4, i//4].set_title(title)
    
    fig.savefig(path)
    plt.close()

def save_loss(Enc_losses, Dec_losses, Dis_losses, path="results/loss.png"):
    fig = plt.figure(figsize=(10,5))
    plt.title("Encoder, Decoder and Discriminator Loss During Training")
    plt.plot(Enc_losses, label="Enc")
    plt.plot(Dec_losses, label="Dec")
    plt.plot(Dis_losses, label="Dis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path)
    plt.close()
