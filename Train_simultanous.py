
import  torch
import torch.utils
import torch.utils.data

import numpy as np
import vae.VAE_KL_Study.utils as utils
from vae.VAE_KL_Study.evaluation import evaluate

def train_sim(cfg,encoder, decoder,train_data,eval_data ):


    optimizer1 = torch.optim.RMSprop(list(encoder.parameters()),
                                     lr=cfg.learning_rate,
                                     centered=True)

    optimizer2 = torch.optim.RMSprop(list(decoder.parameters()),
                                     lr=cfg.learning_rate,
                                     centered=True)
    for epoc in range(cfg.n_epochs):
        best_valid_elbo = -np.inf
        for step, batch in enumerate(utils.cycle(train_data)):
            x = batch[0].to("cpu")

            decoder.zero_grad()
            encoder.zero_grad()
            z, enc_score = encoder(x)

            # average over sample dimension
            optimizer1.zero_grad()
            loss = enc_score.mean(1).sum(0)
            loss.backward()

            optimizer1.step()
            z = torch.tensor(z, requires_grad=False) #We wish Z to be non differntaible
            optimizer2.zero_grad()
            dec_score, _ = decoder(z, x)
            bce_score = dec_score.mean(axis=1).sum(0)

            loss = -bce_score
            loss.backward()
            optimizer2.step()



            with torch.no_grad():

                test_elbo= evaluate(cfg.n_samples, decoder, encoder, eval_data, combined=False)
                if test_elbo > best_valid_elbo:

                    best_valid_elbo = test_elbo
                    states = {'model': decoder.state_dict(),
                              'variational': encoder.state_dict()}
                    torch.save(states, cfg.train_dir + 'best_state_dict')

    print ("final results")
    with torch.no_grad():
        _ = evaluate(cfg.n_samples, decoder, encoder, eval_data, combined=False)