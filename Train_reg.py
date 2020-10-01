
import  torch
import torch.utils
import torch.utils.data

import numpy as np
import vae.VAE_KL_Study.utils as utils
from vae.VAE_KL_Study.evaluation import evaluate
from vae.VAE_KL_Study.loss_manager import Loss_manager

# This train is used for one of the two :
  # Regular training (using ELBO)
  # Training only the cnoder

def train_reg(cfg,encoder, decoder,train_data,eval_data ):


    loss_obj= Loss_manager(cfg.regular)
    if cfg.regular:
        optimizer  = torch.optim.RMSprop(list(encoder.parameters())+list(decoder.parameters()),
                                     lr=cfg.learning_rate,
                                     centered=True)

    else:
        optimizer = torch.optim.RMSprop(list(encoder.parameters()),
                                        lr=cfg.learning_rate,
                                        centered=True)

    for epoc in range(cfg.n_epochs):
        best_valid_elbo = -np.inf
        for step, batch in enumerate(utils.cycle(train_data)):
            x = batch[0].to("cpu")

            decoder.zero_grad()
            encoder.zero_grad()
            optimizer.zero_grad()
            score= loss_obj.tot_los(x,encoder,decoder)

            loss = score.mean(1).sum(0)


            loss.backward()
            optimizer.step()

            with torch.no_grad():

                test_elbo= evaluate(cfg.n_samples, decoder, encoder, eval_data, combined=cfg.regular)
                if test_elbo > best_valid_elbo:

                    best_valid_elbo = test_elbo
                    states = {'model': decoder.state_dict(),
                              'variational': encoder.state_dict()}
                    torch.save(states, cfg.train_dir + 'ggggg')

    print ("final results")
    with torch.no_grad():
        _ = evaluate(cfg.n_samples, decoder, encoder, eval_data, combined=False)