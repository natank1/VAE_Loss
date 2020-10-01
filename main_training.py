import torch
import numpy as np
import random
import vae.VAE_KL_Study.utils as utils
from vae.VAE_KL_Study.encoder_mean_field import EncoderMF
from vae.VAE_KL_Study.decoder import Decoder
from vae.VAE_KL_Study.Train_reg import  train_reg
from vae.VAE_KL_Study.Train_simultanous import train_sim
from vae.VAE_KL_Study.get_mnist_data import load_binary_mnist
import h5py

#
# #This function was brought from here
# # https://github.com/altosaar/variational-autoencoder/blob/master/train_variational_autoencoder_pytorch.py
# def load_binary_mnist(cfg, **kwcfg):
#     fname = cfg.data_dir +"fileall.hdf5"
#     print (fname)
#     # if not fname.exists():
#     #     print('Downloading binary MNIST data...')
#     #     data.download_binary_mnist(fname)
#     # f = h5py.File(pathlib.os.path.join(pathlib.os.environ['DAT'], "fileall.hdf5"), 'r')
#     f=h5py.File(fname,'r')
#     x_train = f['train'][::]
#     x_val = f['valid'][::]
#     x_test = f['test'][::]
#     train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
#     train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True, **kwcfg)
#     validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
#     val_loader = torch.utils.data.DataLoader(validation, batch_size=cfg.test_batch_size, shuffle=False)
#     test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
#     test_loader = torch.utils.data.DataLoader(test, batch_size=cfg.test_batch_size, shuffle=False)
#     return train_loader, val_loader, test_loader







if __name__ == '__main__':
    cfg =utils.create_cfg()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    decoder = Decoder(latent_size=cfg.latent_size, data_size=cfg.data_size)
    device="cpu"
    decoder.to(device)
    encoder = EncoderMF(cfg)
    encoder.to(device)
    kwargs = {}
    train_data, valid_data, test_data = load_binary_mnist(cfg, **kwargs)
    if cfg.regular or not(cfg.simul):
        train_reg(cfg, encoder, decoder, train_data, test_data)
    else:

        train_sim(cfg, encoder, decoder, train_data, test_data)