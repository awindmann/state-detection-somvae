import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim=300, h1_dim=128, h2_dim=32, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()  # stores hyperparameters in self.hparams and allows logging

        self.encoder = nn.Sequential(nn.Linear(self.hparams.input_dim, self.hparams.h1_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hparams.h1_dim, self.hparams.h2_dim))
        self.decoder = nn.Sequential(nn.Linear(self.hparams.h2_dim, self.hparams.h1_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hparams.h1_dim, self.hparams.input_dim))

    def _shared_step(self, x):
        """Shared step used in training, validation and test."""
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss_fct = nn.MSELoss()
        loss = loss_fct(x, x_hat)
        return loss

    def training_step(self, batch, batch_id):
        loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_id):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        # log hparams with val_loss as reference
        if self.logger:
            self.logger.log_hyperparams(self.hparams, {"hp/val_loss": torch.min(torch.stack(outputs))})

    def test_step(self, batch, batch_id):
        loss = self._shared_step(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}]
