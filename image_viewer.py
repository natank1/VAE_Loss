from vae.VAE_KL_Study.decoder import Decoder
from vae.VAE_KL_Study.encoder_mean_field import EncoderMF
import  vae.VAE_KL_Study.utils as utils
from vae.VAE_KL_Study.get_mnist_data import load_binary_mnist
import torch




if __name__=='__main__':
    model_test="C:\\tt\\vae_trial\second_v\\after_fix0_aa_best_state_dict_False_128_False_0.0_64"
    print ("aa")
    cfg = utils.create_cfg()
    decoder = Decoder(latent_size=128, data_size=784)
    encoder = EncoderMF(cfg)

    decoder.load_state_dict(torch.load(model_test)['model'])
    encoder.load_state_dict(torch.load(model_test)['variational'])

    with torch.no_grad():
        _, _, test_data= load_binary_mnist(cfg, **{})
        for batch in test_data:
            x = batch[0].to(next(decoder.parameters()).device)

            #

            z, _  = encoder(x)
            _, logits = decoder(z, x)
            utils.generate_images(logits, cfg)


