import torch
from torch import nn
from vae.VAE_KL_Study.utils import NeuralNetwork,NormalLogProb
from vae.VAE_KL_Study.gauss_kl import kl_div_std_gauss_torch as kl_anal
from vae.VAE_KL_Study.loss_manager import Loss_manager as loss_m

class EncoderMF(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, cfg):
        super().__init__()
        self.inference_network = NeuralNetwork(input_size=cfg.data_size,
                                               output_size=cfg.latent_size * 2,
                                               hidden_size=cfg.latent_size * 2)


        self.log_p= NormalLogProb()

        self.softplus = nn.Softplus()
        self.register_buffer('p_z_loc', torch.zeros(cfg.latent_size))
        self.register_buffer('p_z_scale', torch.ones(cfg.latent_size))
        self.n_samples = cfg.n_samples
        if cfg.use_anal:
            self.encoder_score = self.analytic_score
        else:
            self.encoder_score = self.mc_estimator


    def get_z(self,x ):
        x = x.type(torch.FloatTensor)
        loc, scale_arg = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], self.n_samples, loc.shape[-1]))
        z = loc + scale * eps  # reparameterization
        return z,loc, scale

    def mc_estimator(self, loc, scale, z):
        log_q_z = self.log_p(loc, scale, z).sum(-1, keepdim=True)
        log_p_z = self.log_p(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
        kl_measure = log_q_z - log_p_z
        return kl_measure

    def analytic_score(self, loc, scale, z):
        anal_kl = kl_anal(loc, scale)
        return anal_kl

    def forward(self, x):
        """Return sample of latent variable and log prob."""

        z,loc,scale =self.get_z(x )
        score_enc= self.encoder_score(loc,scale,z)

        return z, score_enc