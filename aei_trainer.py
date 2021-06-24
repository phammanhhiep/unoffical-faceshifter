import os
import logging
from datetime import datetime

from omegaconf import OmegaConf
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from aei_net import AEINet


def main(args):
    hp = OmegaConf.load(args.config)
    model = AEINet(hp)
    save_pth = os.path.join(hp.chkpt.root_dir, hp.model.experiment)
    os.makedirs(save_pth, exist_ok=True)

    if hp.log.to_file:
        logger = logging.getLogger("pytorch_lightning")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.FileHandler(
            "artifacts/log/pytorch_lightning_{}.log".format(
                datetime.today().strftime('%Y%m%d_%H%M%S'))))        

    pl_logger = TensorBoardLogger(hp.log.root_dir, 
        name=hp.model.experiment,
        version=hp.log.version,
        default_hp_metric=False,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_pth,
        # filename=hp.model.name + "-{epoch:03d}-{val_loss:.3f}",
        filename=hp.model.name + "-{epoch:03d}",
        monitor=hp.chkpt.monitor,
        save_top_k=hp.chkpt.save_top_k,
        every_n_val_epochs=hp.chkpt.every_n_val_epochs,
        every_n_train_steps=hp.chkpt.every_n_train_steps,
        save_last=hp.chkpt.save_last,
        verbose=True,
    )

    trainer_opt = {
        "logger": pl_logger,
        "callbacks": [checkpoint_callback],
        "num_processes": hp.trainer.num_processes,
        "num_sanity_val_steps": 1,
        "gradient_clip_val": hp.model.grad_clip,
        "val_check_interval": hp.trainer.val_check_interval,
        "progress_bar_refresh_rate": 1,
        "max_epochs": hp.trainer.max_epoch, 
        "limit_val_batches": hp.trainer.limit_val_batches,
        "limit_train_batches": hp.trainer.limit_train_batches, 
        "log_every_n_steps": hp.log.log_every_n_steps,
        "fast_dev_run": args.fast_dev_run,
        "resume_from_checkpoint": args.checkpoint,
    }

    # TODO: check command-line arguments to add gpu options to trainer_opt

    trainer = Trainer(**trainer_opt)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the resumed checkpoint")
    args = parser.parse_args()

    if args.checkpoint == "":
        args.checkpoint = None

    main(args)