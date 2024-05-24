import pathlib
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import open_clip
from torch import nn, optim, Tensor, manual_seed, argmax
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

import torchmetrics
from torchmetrics.classification import Accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(pl.LightningModule):
    def __init__(self, input_size, num_classes=3, xcol='embeddings', ycol='label_id', optimizer="sdg"):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr = {'Adam': 0.001, 'SGD': 0.1}[optimizer]

        # Set the example_input_array for TensorBoard
        self.example_input_array = torch.randn(1, self.input_size)

        print("Input_Size: ", input_size)
        self.xcol = xcol
        self.ycol = ycol
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)  # Output 3 logits for 3 classes
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('loss', loss)

        # Track accuracy
        y_target = argmax(y, dim=1)
        y_pred = argmax(logits, dim=1)
        acc = self.accuracy(y_pred, y_target)
        self.log('accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        print(logits)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)

        # Track accuracy
        y_target = y
        y_pred = argmax(logits, dim=1)
        acc = self.accuracy(y_pred, y_target)
        self.log('val_accuracy', acc, logger=True)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer
        else:
            default_lr = 1e-3
            weight_decay = 1e-5
            epoch_lr_map = {1: 1e-3, 6750: 1e-5}
            current_lr_factor = 1.0

            def lr_lambda(epoch):
                nonlocal current_lr_factor  # Declare the variable as nonlocal to modify it
                if epoch in epoch_lr_map:
                    current_lr_factor = epoch_lr_map[epoch] / default_lr  # Update the factor
                return current_lr_factor  # Return the current factor

            optimizer = torch.optim.SGD(self.parameters(), lr=default_lr, momentum=0.9, weight_decay=weight_decay)
            scheduler = LambdaLR(optimizer, lr_lambda)
            return [optimizer], [scheduler]
        return optimizer


def custom_collate_fn(batch):
    x_ = default_collate(batch)
    return tuple(item.to(device) for item in x_)


def train_predictor(root_folder, database_file, train_from, clip_models, val_percentage=0.25, epochs=5000,
                    batch_size=1000):
    # clip_models=[('ViT-B-16', 'openai'),('RN50', 'openai')]

    prefix = database_file.split(".")[0]
    out_path = pathlib.Path(root_folder)
    x_out = f"{prefix}_x_concatenated_embeddings.npy"
    y_out = f"{prefix}_y_{train_from}.npy"
    save_name = f"{prefix}_linear_predictor_concatenated_{train_from}_mse.pth"

    x_embeddings_path = out_path / x_out
    y_features_path = out_path / y_out
    x_embeddings = np.load(x_embeddings_path)
    y_features = np.load(y_features_path)

    train_border = int(x.shape[0] * (1 - val_percentage))

    train_tensor_x = torch.Tensor(x[:train_border])
    train_tensor_y = torch.Tensor(y[:train_border]).long()
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    val_tensor_x = torch.Tensor(x[train_border:])
    val_tensor_y = torch.Tensor(y[train_border:]).long()
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    total_dim = 0
    for clip_model in clip_models:
        # get_model_config("ViT-B-16-SigLIP-512")["embed_dim"]
        if clip_model[0] == "hf-hub:timm":
            config = open_clip.get_model_config(clip_model[1])  # ["embed_dim"]
            print(f"SigLip model with {config['embed_dim']} dimension")
        else:
            config = open_clip.get_model_config(clip_model[0])

        if config is not None and 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")

    # Use the total dimension for the MLP model
    print("total_dim: ", total_dim)

    model = MLP(total_dim, num_classes=3)

    if clip_model[0] == "hf-hub:timm":
        model_names = "clip_" + clip_model[0].replace(':', '_') + "/" + clip_model[1]
        # Extract model names from clip_models and join them with underscores
    else:
        model_names = "_".join([model[0].replace('/', '').lower() for model in clip_models])

    # Use the combined model names for the logger
    name = f'{model_names}'
    print(f"Tensor Board log name : {name}")
    logger = TensorBoardLogger('../tb_logs', name=name, log_graph=True)

    # Create the LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Add the callback to the Trainer
    trainer = pl.Trainer(
        logger=logger,  # This is your TensorBoardLogger
        max_epochs=epochs,
        devices="auto",
        accelerator="auto",
        callbacks=[lr_monitor],
        check_val_every_n_epoch=int((epochs / batch_size) * 5)  # Check validation every 10 epochs
    )

    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, val_loader)

    # trainer.save_checkpoint(str(out_path / save_name))

    torch.save(model.state_dict(), out_path / save_name)
