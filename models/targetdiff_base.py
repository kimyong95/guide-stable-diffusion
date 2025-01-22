
import torch
import numpy as np
import einops
import os
import math
from models.base import LightningBase
import lightning as L
from tqdm import tqdm
from lightning.pytorch.loggers.logger import DummyLogger
import torchvision
import lightning
import PIL
import inspect
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import asyncio

import random
import shutil
from related_works.targetdiff.utils import misc, reconstruct, transforms
from rdkit import Chem
from related_works.targetdiff.models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from onediff.infer_compiler import oneflow_compile
import related_works.targetdiff.utils.transforms as trans
from related_works.targetdiff.utils.evaluation import scoring_func
from related_works.targetdiff.utils.evaluation.docking_vina import VinaDockingTask, PrepLig
from related_works.targetdiff.datasets import get_dataset
from related_works.targetdiff.scripts.sample_diffusion import sample_diffusion_ligand
from torch_geometric.transforms import Compose
# from DeepCache import DeepCacheSDHelper


class TargetdiffBase(LightningBase):

    def __init__(self, data_id, vina_web_url=None, pos_only=True, scheduler="ddim"):

        super().__init__()

        self.targetdiff_root = "related_works/targetdiff"
        self.scheduler = scheduler
        self.model, self.data = self.get_targetdiff(data_id=data_id)
        self.vina_web_url = vina_web_url
        self.pos_only = pos_only
        self.num_atoms = self.data.ligand_pos.shape[0]
        self.dim = self.data.ligand_pos.shape[1]
        
        if scheduler == "ddim":
            self.num_sampling_steps = 200
        elif scheduler == "guide-eular":
            self.num_sampling_steps = 20
        else:
            raise NotImplementedError()

    
    def get_targetdiff(self, data_id=0, ckpt_path="./pretrained_models/pretrained_diffusion.pt"):
        
        ckpt = torch.load(self.resolve_relative_dir(ckpt_path))

        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
        ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
        transform = Compose([
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ])

        ckpt['config'].data["path"] = self.resolve_relative_dir(ckpt['config'].data["path"])
        ckpt['config'].data["split"] = self.resolve_relative_dir(ckpt['config'].data["split"])

        # Load dataset
        dataset, subsets = get_dataset(
            config=ckpt['config'].data,
            transform=transform
        )
        train_set, test_set = subsets['train'], subsets['test']

        model = ScorePosNet3D(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
            scheduler=self.scheduler,
        )
        model.load_state_dict(ckpt['model'])

        data = test_set[data_id]

        return model, data
    
    def resolve_relative_dir(self, path):
        return os.path.normpath(os.path.join(self.targetdiff_root, path))

    def on_fit_start(self):
        super().on_fit_start()
        self.model = self.model.to(self.device)

        # if not debug mode
        # if not isinstance(self.trainer.logger, DummyLogger):
        #     self.compile()

        # self.model = torch.compile(self.model)
        return self

    def compile(self):
        self.model = oneflow_compile(self.model)
        cache_path = f"onediff_cache/targetdiff"
        try:
            self.model.load_graph(cache_path, device=str(self.device))
        except ValueError:
            os.makedirs("onediff_cache", exist_ok=True)
            _ = self.sampling(1)
            self.model.save_graph(cache_path)

    def sampling(self, batch_size, epsilon=None):

        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            self.model, self.data, batch_size,
            batch_size=batch_size, device=self.device,
            num_steps=self.num_sampling_steps,
            pos_only=self.pos_only,
            center_pos_mode="protein",
            sample_num_atoms="ref",
            epsilon=epsilon,
        )
        
        return pred_pos, pred_v

    # minimize vina affinity score
    # minimize scores
    def get_scores(self, pos_list, v_list):

        # Tang S, Chen R, Lin M, Lin Q, Zhu Y, Ding J, Hu H, Ling M, Wu J. Accelerating AutoDock Vina with GPUs. Molecules. 2022 May 9;27(9):3041. doi: 10.3390/molecules27093041. PMID: 35566391; PMCID: PMC9103882.
        # cite: The AutoDock Vina score for drug-like compounds can reach as low as -11.6 kcal/mol.
        # used for normalizing the score to [0, 1]
        MAX_VINA_SCORE = 11.6

        scores = torch.zeros(len(pos_list), device=self.device)
        failed_count = 0
        loop = asyncio.get_event_loop()

        for i, (pos, v) in enumerate(zip(pos_list, v_list)):
            scores[i], failed_ = loop.run_until_complete(self.get_score(pos,v))
            if failed_:
                failed_count += 1

        return scores, failed_count
    
    def get_scores_parallel(self, pos_list, v_list):
        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(self.get_score(pos,v)) for pos, v in zip(pos_list, v_list)]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        scores, failed = map(list, zip(*results))
        failed_count = sum(failed)
        return torch.tensor(scores, dtype=torch.float32, device=self.device), failed_count

    async def get_score(self, pos, v):

        MAX_VINA_SCORE = 11.6

        assert pos.shape == (self.num_atoms, 3)
        score = 0.0
        mol = self.reconstruct_molecule(pos, v)
        if mol is not None:
            vina_task = VinaDockingTask.from_generated_mol(mol, self.data.ligand_filename, protein_root=self.resolve_relative_dir("data/test_set"), web_dock_url=self.vina_web_url)
            score = (await vina_task.run(mode='score_only', exhaustiveness=16))[0]["affinity"] / MAX_VINA_SCORE
            score = score if score < 0.0 else 0.0
        failed = bool(mol is None)

        return score, failed

    def reconstruct_molecule(self, pos, v):
        # reconstruction
        pred_atom_type = transforms.get_atomic_number_from_index(v, mode="add_aromatic")
        try:
            pred_aromatic = transforms.is_aromatic_from_index(v, mode="add_aromatic")
            mol = reconstruct.reconstruct_from_generated(pos, pred_atom_type, pred_aromatic)
        except KeyboardInterrupt:
            raise
        except reconstruct.MolReconsError:
            return None
        
        return mol

    def _x_flatten(self, x):
        return einops.rearrange(x, '... N D -> ... (N D)', N=self.num_atoms, D=self.dim)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (N D) -> ... N D', N=self.num_atoms, D=self.dim)

    def log_score(self, scores, stage="train"):

        MAX_VINA_SCORE = 11.6
        
        self.log(f"{stage}/score_mean", scores.mean())
        self.log(f"{stage}/raw_score_mean", scores.mean() * MAX_VINA_SCORE)

        self.log(f"{stage}/score_min", scores.min())
        self.log(f"{stage}/raw_score_min", scores.min() * MAX_VINA_SCORE)

        self.log(f"{stage}/score_median", scores.median())
        self.log(f"{stage}/raw_score_median", scores.median() * MAX_VINA_SCORE)

    def log_molecules(self, pos_list, v_list, scores):
        
        molecules_list = []
        ligand_list = []
        receptor_list = []

        for i, (pos, v) in enumerate(zip(pos_list, v_list)):
            mol = self.reconstruct_molecule(pos, v)
            if mol is not None:
                vina_task = VinaDockingTask.from_generated_mol(mol, self.data.ligand_filename, protein_root=self.resolve_relative_dir("data/test_set"))
                ligand_str = PrepLig(vina_task.ligand_str, 'sdf').get_pdbqt()
                receptor_file = vina_task.receptor_path[:-4] + '.pdbqt'

                molecules_list.append(mol)
                ligand_list.append(ligand_str)
                receptor_list.append(receptor_file)
            else:
                molecules_list.append(None)
                ligand_list.append(None)
                receptor_list.append(None)

        _dir = f"{self.trainer.logger.experiment.dir}/{self.global_step}/molecules"
        os.makedirs(_dir, exist_ok=True)

        for i, (mol, ligand_file, receptor_file, score) in enumerate(zip(molecules_list, ligand_list, receptor_list, scores)):
            if mol is not None:
                os.makedirs(f"{_dir}/{i}", exist_ok=True)
                with open(f"{_dir}/{i}/ligand.pdbqt", "w") as f:
                    f.write(ligand_str)
                shutil.copyfile(receptor_file, f"{_dir}/{i}/receptor.pdbqt")
        
        # save scores
        with open(f"{_dir}/scores.txt", "w") as f:
            for score in scores:
                f.write(f"{score:.3f}\n")
            