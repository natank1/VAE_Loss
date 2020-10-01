import torch
import numpy as np
import  vae.VAE_KL_Study.utils as utils
from vae.VAE_KL_Study.get_mnist_data import load_binary_mnist
from vae.VAE_KL_Study.decoder import Decoder
from vae.VAE_KL_Study.encoder_mean_field import EncoderMF
from vae.VAE_KL_Study.evaluation import evaluate

def stand_alone_train(train_data,eval_data, cfg, get_z_f, decoder):
    encoder.eval()
    optimizer = torch.optim.RMSprop(list(decoder.parameters()),
                                    lr=cfg.learning_rate,
                                    centered=True)

    for epoch in range(cfg.n_epochs):
        best_valid_elbo = -np.inf
        for step, batch in enumerate(utils.cycle(train_data)):

            x = batch[0].to("cpu")

            decoder.zero_grad()
            if cfg.rnd_mode:
                z = torch.randn(cfg.batch_size, cfg.n_samples, cfg.latent_size)
            else:
                z, _, _ = encoder.get_z(x)

            log_px_given, _ = decoder(z, x)

            # average over sample dimension

            # sum over batch dimension
            loss = -(log_px_given.mean(1).sum(0))
            loss.backward()
            optimizer.step()
            print("loss=", loss)


            with torch.no_grad():

                test_elbo= evaluate(cfg.n_samples, decoder, encoder, eval_data, combined=cfg.regular)
                if test_elbo > best_valid_elbo:

                    best_valid_elbo = test_elbo
                    states = {'model': decoder.state_dict(),
                              'variational': encoder.state_dict()}
                    torch.save(states, cfg.train_dir + 'ggggg')

    print ("Finally")
    with torch.no_grad():

        test_elbo = evaluate(cfg.n_samples, decoder, encoder, eval_data, combined=cfg.regular)
        if test_elbo > best_valid_elbo:
            best_valid_elbo = test_elbo
            states = {'model': decoder.state_dict(),
                      'variational': encoder.state_dict()}
            torch.save(states, cfg.train_dir + 'gg111ggg')


if __name__=="__main__":
    model_test = "C:\\tt\\vae_trial\second_v\\ggggg"

    cfg = utils.create_cfg()
    decoder = Decoder(latent_size=cfg.latent_size, data_size=cfg.data_size)
    encoder = EncoderMF(cfg)

    decoder.load_state_dict(torch.load(model_test)['model'])
    encoder.load_state_dict(torch.load(model_test)['variational'])
    train_data_, _, test_data = load_binary_mnist(cfg, **{})
    stand_alone_train(train_data_, test_data, cfg, encoder, decoder)