# Torch imports
from torch.utils.data import DataLoader

# Custom imports
import torch
import heuristics
from dataset_utils import PathPlanningDataset
from display import Displayer
from astar_refactored import Astar
from custom_types import onehottensor2node2d, nodelist2otensor
from training_module import NeuralAstarTrainingModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

initial_dataset = PathPlanningDataset(
    "/workspaces/neural-Astar/datasets/raw/street",
    16,
    1,
    heuristics.euclidian,
    randomize_points=False,
    max_astar_iterations=10000,
)

dataloader = DataLoader(initial_dataset, 1, False)

checkpoint_callback = ModelCheckpoint(
    monitor="metrics/h_mean", save_weights_only=True, mode="max"
)


training_module = NeuralAstarTrainingModule()
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=1,
    default_root_dir=".",
    max_epochs=100,
    callbacks=[checkpoint_callback],
)

trainer.fit(training_module, dataloader)
