from vae.VAE_KL_Study.gauss_kl import kl_div_std_gauss_torch as kl_anal
from vae.VAE_KL_Study.utils import NormalLogProb

class Loss_manager():
    def __init__(self,regular):
        super().__init__()

        if regular:
            self.tot_los = self.elbo_loss
        else:
            self.tot_los = self.only_encdoer



    # def mc_estimator(self,loc,scale,z):
    #     log_q_z = self.log_p(loc, scale, z).sum(-1, keepdim=True)
    #     log_p_z = self.log_p(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
    #     kl_measure = log_p_z - log_q_z
    #
    #     return kl_measure
    #
    # def analytic_score(self,loc,scale,z):
    #     anal_kl = kl_anal(loc, scale)
    #     return anal_kl

    def elbo_loss(self,x,encoder, decoder):
        z,score_enc=encoder(x)
        score_dec, _= decoder(z,x)
        # print (score_dec.mean(1).sum(0))
        # print(score_enc.mean(1).sum(0))

        return score_enc-score_dec

    def only_encdoer(self,x,encoder, decoder ):
        z,score_enc=encoder(x)

        return score_enc