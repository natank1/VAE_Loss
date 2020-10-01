import torch
import numpy as np
import  vae.VAE_KL_Study.utils as utils
from vae.VAE_KL_Study.get_mnist_data import load_binary_mnist
from vae.VAE_KL_Study.decoder import Decoder
from vae.VAE_KL_Study.encoder_mean_field import EncoderMF
def evaluate( n_samples, decoder,encoder, eval_data,combined=True):

    total_log_p_x = 0.0
    total_enc = 0.0
    total_dec=0.0
    for batch in eval_data:
        x = batch[0].to(next(decoder.parameters()).device)

        z, enc_score  = encoder(x)
        dec_score,_ = decoder(z, x)
        total_dec+=  dec_score.cpu().numpy().mean(1).sum(0)
        total_enc += enc_score.cpu().numpy().mean(1).sum(0)


        log_p_x = torch.logsumexp(enc_score, dim=1) - np.log(n_samples)
        # average over sample dimension, sum over minibatch

        # sum over minibatch
        total_log_p_x += log_p_x.cpu().numpy().sum()
    n_data = len(eval_data.dataset)


    # tot_logpx =total_dec.cpu().numpy()[0]
    if combined :
        test_elbo = total_enc-total_dec
    else:
        test_elbo= total_enc

    # print(f'\ttest elbo: {test_elbo:.2f}\tenc_score log p(x): {total_enc:.2f}\tdec_score :{tot_logpx:.3f}')
    print(f'test elbo: {test_elbo[0]/n_data:.2f}\tenc_score log p(x): {total_enc[0]/n_data:.2f}\tdec_score :{total_dec[0]/n_data:.3f}')
    return test_elbo[0] / n_data


if __name__=='__main__':
    model_test="C:\\tt\\vae_trial\second_v\\after_fix0_aa_best_state_dict_False_128_False_0.0_64"

    cfg = utils.create_cfg()
    decoder = Decoder(latent_size=cfg.latent_size, data_size=cfg.data_size)
    encoder = EncoderMF(cfg)

    decoder.load_state_dict(torch.load(model_test)['model'])
    encoder.load_state_dict(torch.load(model_test)['variational'])
    with torch.no_grad():
        _, _, test_data = load_binary_mnist(cfg, **{})

        evaluate(cfg.n_samples, decoder, encoder, test_data, combined=True)

