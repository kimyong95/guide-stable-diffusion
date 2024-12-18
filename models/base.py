
from typing import Any, Mapping
import lightning as L
import yaml
import importlib
import torch
from lightning.pytorch.loggers.logger import DummyLogger
import matplotlib.pyplot as plt
from utils.utils import disable_train
torch.set_warn_always(False)
plt.rcParams.update({'axes.formatter.limits': (-3, 3)})
import os
class LightningBase(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.ignored_checkpoint_keys = []
        self.replace_checkpoint_keys = []

    def on_fit_start(self):
        log_dir = str(self.trainer.logger.experiment.dir).removesuffix("/files")
        ckpt_dir = os.path.realpath(os.path.expanduser(f"{log_dir}/checkpoints"))
        self.trainer.callbacks[-1].dirpath = f"{log_dir}/checkpoints"

        for i in range(len(self.trainer.callbacks)):
            if type(self.trainer.callbacks[i]) == L.pytorch.callbacks.ModelCheckpoint:
                self.trainer.callbacks[i].dirpath = ckpt_dir
        
        super().on_fit_start()
        
    @staticmethod
    def load_from_run_path(run_path, checkpoint="last", training=False):
        checkpoint_path = f"{run_path}/checkpoints/{checkpoint}.ckpt"
        config_path = f"{run_path}/files/config-cli.yaml"
        with open(config_path, 'r') as f:
            cli_config = yaml.safe_load(f)

        datamodule = LightningBase.__instantiate_from_config(cli_config["data"])
        model = LightningBase.__instantiate_from_config(cli_config["model"])
        model.datamodule = datamodule

        datamodule.setup("predict")
        model.setup("predict")

        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        
        if not training:
            model = disable_train(model)
        
        return model

    @staticmethod
    def __instantiate_from_config(subclass_config):
        class_path = subclass_config["class_path"]
        init_args = subclass_config["init_args"]
        module_str, class_str = class_path.rsplit(".",1)
        Cls = getattr(importlib.import_module(module_str), class_str)
        instance = Cls(**init_args)
        return instance

    def on_load_checkpoint(self, checkpoint):

        # add back the ignored keys during on_save_checkpoint
        deleted_keys = []
        for k in self.state_dict().keys():
            if any([k.startswith(i) for i in self.ignored_checkpoint_keys]):
                deleted_keys.append(k)
        
        for k in deleted_keys:
            checkpoint["state_dict"][k] = self.state_dict()[k]

        for k in self.replace_checkpoint_keys:
            setattr(self, k, checkpoint["state_dict"][k])

    def on_save_checkpoint(self, checkpoint):

        delete_keys = []
        for k in checkpoint["state_dict"].keys():
            if any([k.startswith(i) for i in self.ignored_checkpoint_keys]):
                delete_keys.append(k)
        
        for k in delete_keys:
            checkpoint["state_dict"].pop(k)