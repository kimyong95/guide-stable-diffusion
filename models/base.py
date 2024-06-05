
from typing import Any, Mapping
import lightning as L
import yaml
import importlib
import torch

class LightningBase(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.ignored_checkpoint_keys = []


    @staticmethod
    def disabled_train_func(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

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
            model.eval()
            model.train = LightningBase.disabled_train_func
            model.requires_grad_(False)
        
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

    def on_save_checkpoint(self, checkpoint):

        delete_keys = []
        for k in checkpoint["state_dict"].keys():
            if any([k.startswith(i) for i in self.ignored_checkpoint_keys]):
                delete_keys.append(k)
        
        for k in delete_keys:
            checkpoint["state_dict"].pop(k)