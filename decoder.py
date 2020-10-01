import  torch
import torch.utils
import torch.utils.data
from vae.VAE_KL_Study.utils import NeuralNetwork, BernoulliLogProb, cycle


from torch import nn

import numpy as np
class Decoder(nn.Module):
    """Bernoulli model parameterized by a generative network with Gaussian latents for MNIST."""

    def __init__(self, latent_size, data_size):
        super().__init__()

        self.log_p_x = BernoulliLogProb()
        self.generative_network = NeuralNetwork(input_size=latent_size,
                                                output_size=data_size,
                                                hidden_size=latent_size * 2)

    def get_p_given_z_score(self ,logits ,x):
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return  log_p_x


    def forward(self, z, x):

        logits = self.generative_network(z)
        log_p_x = self.get_p_given_z_score(logits ,x)

        return  log_p_x,logits




    def stand_alone_train(train_data,cfg,get_z_f,decoder):
        optimizer = torch.optim.RMSprop(list(decoder.parameters()),
                                        lr=cfg.learning_rate,
                                        centered=True)

        for epoch in range (cfg.n_epoch):
            best_valid_elbo = -np.inf
            for step, batch in enumerate(cycle(train_data)):


                    x = batch[0].to("cpu")

                    decoder.zero_grad()
                    if cfg.rnd_mode:
                       z =torch.randn(cfg.batch_size,cfg.n_samples,cfg.latent_size)
                    else:
                        z,_,_= get_z_f(x, n_samples=cfg.n_samples)

                    log_px_given,_ = decoder(z, x)

                    # average over sample dimension

                    # sum over batch dimension
                    loss = -(log_px_given.mean(1).sum(0))
                    loss.backward()
                    optimizer.step()
                    print ("loss=",loss)

