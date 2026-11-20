import torch
import torch.nn.functional as F

__all__ = ["Losses"]


class Losses:
    def __init__(self, vae_input_dim, class_weights, sequence=True,
                 bce_weight=2.0, mse_weight=0.5, kld_weight=0.5):
        self.vae_input_dim = vae_input_dim
        self.sequence = sequence
        # Loss weights on the prediction/BCE, reconstruction MSE, and KL terms.
        # Defaults match the paper (lambda_pred=2.0 / lambda_recon=0.5 /
        # lambda_normal=0.5); the original upstream BCE_loss used 5.0/0.1/0.1.
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.kld_weight = kld_weight

        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        self.pos_weight = torch.tensor(float(class_weights[0]) / float(class_weights[1])).float()

    def regression_loss(self, recon_x, x, mu, logvar, final_output, y):
        regression = F.mse_loss(final_output.squeeze(), y.squeeze(), reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return self.bce_weight * regression + self.mse_weight * MSE + self.kld_weight * KLD
        else:
            return regression

    def BCE_loss(self, recon_x, x, mu, logvar, final_output, y):
        BCE = F.binary_cross_entropy_with_logits(final_output.view(-1), y.view(-1), pos_weight=self.pos_weight, reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return self.bce_weight * BCE + self.mse_weight * MSE + self.kld_weight * KLD
        else:
            return BCE

    def regression_loss_SSL(self, recon_x, x, mu, logvar, final_output, y, pred_amino_acid, amino_acid):
        amino_classification = 0

        if pred_amino_acid.numel():
            amino_classification = F.cross_entropy(pred_amino_acid, amino_acid)

        regression = F.mse_loss(final_output.squeeze(), y.squeeze(), reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return self.bce_weight * regression + self.mse_weight * MSE + self.kld_weight * KLD + amino_classification
        else:
            return regression + amino_classification

    def BCE_loss_SSL(self, recon_x, x, mu, logvar, final_output, y, pred_amino_acid, amino_acid):
        amino_classification = 0

        if pred_amino_acid.numel():
            amino_classification = F.cross_entropy(pred_amino_acid, amino_acid)

        BCE = F.binary_cross_entropy_with_logits(final_output.view(-1), y.view(-1), pos_weight=self.pos_weight, reduction='mean')

        if self.sequence:
            MSE = F.mse_loss(recon_x, x.view(-1, self.vae_input_dim), reduction='mean')
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return self.bce_weight * BCE + self.mse_weight * MSE + self.kld_weight * KLD + amino_classification
        else:
            return BCE + amino_classification
