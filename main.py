# main.py
from typing import Any, Callable, Dict, Optional, Type, Union
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import ArgsType, LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from jsonargparse import lazy_instance
from lightning.pytorch.loggers import WandbLogger
import lightning
import os
import wandb
import inspect
import shutil
import torch
torch.set_float32_matmul_precision('high')

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

# Extend to save to wandb dir

def get_all_base_classes(class_type: type):

    if class_type in [lightning.LightningModule, lightning.LightningDataModule]:
        return []
    
    base_classes = [class_type]
    for base_class in class_type.__bases__:
        base_classes.extend(get_all_base_classes(base_class))
    return base_classes

class WandbSaveConfigCallback(SaveConfigCallback):

    def __init__(self, *args, **kwargs):
        kwargs["config_filename"] = "config-cli.yaml"
        kwargs["save_to_log_dir"] = False
        super().__init__(*args, **kwargs)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            config_path = os.path.join(trainer.logger.experiment.dir, self.config_filename)
            self.parser.save(self.config, config_path)

            log_classes = get_all_base_classes(trainer.model.__class__) + get_all_base_classes(trainer.datamodule.__class__)

            log_codes = [inspect.getfile(cls) for cls in log_classes]
            
            trainer.logger.experiment.log_code(".", include_fn=lambda path: path in log_codes)

            for code in log_codes:
                rel_code = os.path.relpath(code)
                dest = trainer.logger.experiment.dir + f"/code/{rel_code}"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copyfile(code, dest)
                    
class WandbModelCheckpoint(ModelCheckpoint):

    def _ModelCheckpoint__resolve_ckpt_dir(self, trainer):
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
            log_dir = str(trainer.logger.experiment.dir).rstrip("/files")
            return log_dir + "/checkpoints"
        else:
            return super()._ModelCheckpoint__resolve_ckpt_dir(trainer)

def cli_main():
    cli = LightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=WandbSaveConfigCallback,
    )

if __name__ == "__main__":
    cli_main()