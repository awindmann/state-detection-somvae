import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from data.datamodule import ThreeTankStateDataModule
from models.som_vae.som_vae import SOMVAE


def train(config, hparams=None):
    pl.seed_everything(1234)
    dm = ThreeTankStateDataModule(
        nb_of_samples=10000,
        window_size=100,
        ordered_samples=False,
        batch_size=config.pop("batch_size"),
        num_workers=config.pop("num_workers"),
        pin_memory=False)

    # trainer callbacks
    ckpt_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}')
    lr_monitor = LearningRateMonitor()
    early_stop = EarlyStopping(monitor="val_loss", patience=15)
    logger = TensorBoardLogger(config.pop("logdir"), name=config.pop("model_name"), default_hp_metric=False)

    trainer = pl.Trainer(**config,
                         callbacks=[ckpt_callback, lr_monitor, early_stop],
                         logger=logger)

    # train
    model = SOMVAE(**hparams)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    config = dict(gpus=1,
                  num_workers=8,
                  batch_size=32,
                  max_epochs=150,
                  # ckpt_path=glob.glob("../logs/SOMVAE/final/version_0/checkpoints/*.ckpt")[0],
                  logdir="../logs/SOMVAE",
                  model_name="final")
    hparams = dict(d_som=[2, 3],
                   d_latent=64,
                   d_enc_dec=100,
                   alpha=0.75,
                   beta=2,
                   gamma=0,
                   tau=0)

    train(config, hparams)

