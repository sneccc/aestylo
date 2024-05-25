import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
import pathlib
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from typing import Optional
import open_clip
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torchmetrics import F1Score, Precision, Recall
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, input_size,num_classes, hidden_units=(1024,512, 256, 128)):  # class_weights=None
        super().__init__()
        # self.test_acc = Accuracy()

        # Loss function
        # self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

        # Train Metrics
        self.train_acc = Accuracy(num_classes=num_classes, average='macro', multiclass=True)
        self.train_f1_score = F1Score(num_classes=num_classes, average='macro', multiclass=True)
        self.train_precision = Precision(num_classes=num_classes, average='macro', multiclass=True)
        self.train_recall = Recall(num_classes=num_classes, average='macro', multiclass=True)

        # Validation Metrics
        self.val_acc = Accuracy(num_classes=num_classes, average='macro', multiclass=True)
        self.val_f1_score = F1Score(num_classes=num_classes, average='macro', multiclass=True)
        self.val_precision = Precision(num_classes=num_classes, average='macro', multiclass=True)
        self.val_recall = Recall(num_classes=num_classes, average='macro', multiclass=True)

        all_layers = [nn.Flatten()]
        for index, hidden_unit in enumerate(hidden_units):
            all_layers.append(nn.Linear(input_size, hidden_unit))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(0.5))  # Reduced dropout
            input_size = hidden_unit

        # Output layer for 3 classes (assuming classification with 3 exclusive classes)
        all_layers.append(nn.Linear(hidden_units[-1], num_classes))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    # def training_step_old(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = nn.functional.cross_entropy(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.train_acc.update(preds, y)
    #     self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
    #     return loss
    # def validation_step_old(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = nn.functional.cross_entropy(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.valid_acc.update(preds, y)
    #     self.validation_step_outputs.append(loss)
    #     self.log("valid_loss", loss, prog_bar=True, on_step=False,
    #              on_epoch=True)
    #     return {"val_loss": loss, "preds": preds, "targets": y}

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, y)

        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        self.val_f1_score.update(logits, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(logits, y)
        self.val_acc.update(preds, y)

        return loss

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        self.train_f1_score.update(logits, y)
        self.train_precision.update(preds, y)
        self.train_recall.update(logits, y)
        self.train_acc.update(preds, y)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.log('train_f1_score', self.train_f1_score.compute(), prog_bar=False)
        self.log('train_precision', self.train_precision.compute())
        self.log('train_recall', self.train_recall.compute(), prog_bar=False)

        # Reset metrics at the end of each epoch
        self.train_acc.reset()
        self.train_f1_score.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute())
        self.log('val_f1_score', self.val_f1_score.compute(), prog_bar=False)
        self.log('val_precision', self.val_precision.compute(), prog_bar=False)
        self.log('val_recall', self.val_recall.compute(), prog_bar=False)

        # Reset metrics at the end of each epoch
        self.val_acc.reset()
        self.val_f1_score.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = nn.functional.cross_entropy(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.test_acc.update(preds, y)
    #     self.log("test_loss", loss, prog_bar=True)
    #     self.log("test_acc", self.test_acc.compute(), prog_bar=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = ("sgd")

        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
            return optimizer
        elif optimizer == "warmup":
            warmup_epochs = 1000
            base_lr = 0.001
            max_lr = 0.01
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

            def lr_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    # Linear warm-up from base_lr to max_lr
                    return (max_lr / base_lr) * (current_epoch / warmup_epochs)
                else:
                    # Post warm-up: you can define decay or constant rate here
                    return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return [optimizer], [scheduler]
        else:
            #5e-5 was good
            # Initial learning rate
            default_lr = 3e-4

            # Learning rate adjustments for specific epochs
            epoch_lr_map = {1: 3e-4}  # Adjust learning rate at epoch 1

            # Define a lambda function for learning rate scheduling
            lr_lambda = lambda epoch: epoch_lr_map.get(epoch, default_lr) / default_lr

            # Initialize the optimizer with weight decay
            optimizer = torch.optim.SGD(self.parameters(), lr=default_lr, momentum=0.9, weight_decay=1e-8)

            # Set up the learning rate scheduler
            scheduler = LambdaLR(optimizer, lr_lambda)

            return [optimizer], [scheduler]


def start_training(root_folder, database_file, train_from, clip_models, val_percentage=0.25, epochs=10000,
                   batch_size=1000):
    train_dataloader, val_dataloader, model_name, class_weight,num_classes = setup_dataset(root_folder=root_folder,
                                                                               database_file=database_file,
                                                                               train_from=train_from)
    input_size = get_total_dim(clip_models)
    print("input size", input_size)  # 1152

    net = MultiLayerPerceptron(input_size=input_size,num_classes=num_classes)
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode='max', monitor="val_acc"),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            verbose=True,
            mode='min'
        )
    ]  # save top 1 model
    logger = TensorBoardLogger('tb_logs', name="my_logger", log_graph=True)
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,  # This is your TensorBoardLogger
        max_epochs=epochs,
        devices="auto",
        accelerator="gpu",
        callbacks=callbacks,
        check_val_every_n_epoch=10  # Check validation every 10 epochs
    )

    trainer.fit(net, train_dataloader, val_dataloader)

    save_path = pathlib.Path(root_folder) / model_name
    print("-> saving model in to.. ", save_path)
    torch.save(net.state_dict(), save_path)


def custom_collate_fn(batch):
    x_ = default_collate(batch)
    return tuple(item.to(device) for item in x_)


def setup_dataset(root_folder, database_file, train_from):
    prefix = database_file.split(".")[0]
    out_path = pathlib.Path(root_folder)
    x_out = f"{prefix}_x_concatenated_embeddings.npy"
    y_out = f"{prefix}_y_{train_from}.npy"
    model_name = f"{prefix}_linear_predictor_concatenated_{train_from}_mse.pth"

    x_embeddings_path = out_path / x_out
    y_features_path = out_path / y_out
    x_embeddings = np.load(x_embeddings_path)
    y_features = np.load(y_features_path)

    validation_percentage = 0.15
    batch_size = 256

    indices = np.arange(x_embeddings.shape[0])
    np.random.shuffle(indices)

    x_embeddings = x_embeddings[indices]
    y_features = y_features[indices]

    x_train, x_val, y_train, y_val = train_test_split(
        x_embeddings, y_features, test_size=validation_percentage, random_state=42, stratify=y_features
    )

    train_tensor_x = torch.Tensor(x_train)
    train_tensor_y = torch.Tensor(y_train).long()
    val_tensor_x = torch.Tensor(x_val)
    val_tensor_y = torch.Tensor(y_val).long()

    # Calculate class distribution for weights
    class_counts = np.bincount(train_tensor_y.numpy())
    print("🐍 Class Frequency: ", class_counts)
    total_samples = len(train_tensor_y)
    num_classes = len(class_counts)
    print("🐍 Number of classes: ", num_classes)
    class_weights = total_samples / (num_classes * class_counts)
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("🐍 Class weights: ", class_weight_tensor)


    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, model_name, class_weight_tensor.to(device),num_classes


def get_total_dim(clip_models):
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
    return total_dim
