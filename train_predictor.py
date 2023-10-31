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
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import torchmetrics
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


torch.manual_seed(42)
np.random.seed(42)


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        # Set the example_input_array for TensorBoard
        self.example_input_array = torch.randn(1, self.input_size)
        
        print("Input_Size: ",input_size)
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        

    def forward(self, x):
        #x = self.layers(x)
        # Apply sigmoid and scale to (1, 2) range
        #return 1 + torch.sigmoid(x)
        #return self.layers(x)
        x = self.layers(x)
        return torch.sigmoid(x)  # Squash output between 0 and 1

    def training_step(self, batch, batch_idx):
        #x, y = batch
        #x_hat = self(x)
        #loss = F.mse_loss(x_hat, y)
        #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #return loss
        x, y = batch
        y = (y - 1).float()  # Convert labels from [1, 2] to [0, 1]
        x_hat = self(x)
        loss = F.binary_cross_entropy(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        #y = y.reshape(-1, 1)  # Reshape y to ensure it's a 2D tensor
        #y = (y - 1).float()  # Convert labels from [1, 2] to [0, 1]
        y_binary = (y.cpu().numpy() - 1).astype(int)  # Convert labels from [1, 2] to [0, 1]
        y_binary_tensor = torch.tensor(y_binary).float().to(y.device)
        x_hat = self(x)
        #loss = F.mse_loss(x_hat, y)
        loss = F.binary_cross_entropy(x_hat, y_binary_tensor)


        # Convert continuous predictions to binary labels
        #pred_labels = torch.where(x_hat > 1.5, torch.tensor(2.0).to(x_hat.device), torch.tensor(1.0).to(x_hat.device))
        pred_labels_binary = (x_hat > 0.5).float().cpu().numpy().astype(int)


        
        # Compute F1 score
        #f1 = f1_score(y.cpu().numpy(), pred_labels.cpu().numpy(), pos_label=2)  # considering "good" as the positive class
        # Compute F1 score
        #f1 = f1_score(y.cpu().numpy() * 2 + 1, pred_labels.cpu().numpy(), pos_label=2)  # considering "good" as the positive class
        f1 = f1_score(y_binary, pred_labels_binary, labels=[0, 1], pos_label=1)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_f1': f1}

    def configure_optimizers(self):
            weight_decay = 1e-8# adjust this value as needed
            #optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=weight_decay)
            optimizer = torch.optim.SGD(self.parameters(), lr=2e-3, momentum=0.9, weight_decay=weight_decay)  # Added momentum

            # Define the CosineAnnealingLR scheduler
            #scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-7)  # adjust T_max and eta_min as needed
            #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=500, verbose=True)
            scheduler = {'scheduler':  ReduceLROnPlateau(optimizer, 'min', factor=0.9,cooldown=50, patience=100, verbose=True),'monitor': 'train_loss_epoch' } # or whatever metric you want to monitor
    
            return [optimizer], [scheduler]


def custom_collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_ = default_collate(batch)
    return tuple(item.to(device) for item in x_)


def train_predictor(root_folder, database_file, train_from, clip_models, val_percentage=0.1, epochs=5000, batch_size=1000):

    #clip_models=[('ViT-B-16', 'openai'),('RN50', 'openai')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prefix = database_file.split(".")[0]
    out_path = pathlib.Path(root_folder)
    x_out = f"{prefix}_x_concatenated_embeddings.npy"
    y_out = f"{prefix}_y_{train_from}.npy"
    save_name = f"{prefix}_linear_predictor_concatenated_{train_from}_mse.pth"
    
    x = np.load(out_path / x_out)
    y = np.load(out_path / y_out)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   


    train_border = int(x.shape[0] * (1 - val_percentage))

    train_tensor_x = torch.Tensor(x[:train_border])
    train_tensor_y = torch.Tensor(y[:train_border])
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate_fn)

    val_tensor_x = torch.Tensor(x[train_border:])
    val_tensor_y = torch.Tensor(y[train_border:])
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False ,collate_fn=custom_collate_fn)

 


    total_dim = 0
    for clip_model in clip_models:
        #get_model_config("ViT-B-16-SigLIP-512")["embed_dim"]
        if clip_model[0]== "hf-hub:timm":
            config=open_clip.get_model_config(clip_model[1])#["embed_dim"]
            print(f"SigLip model with {config['embed_dim']} dimension")
        else:
            config = open_clip.get_model_config(clip_model[0])
       

        if config is not None and 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")

    # Use the total dimension for the MLP model
    print("total_dim: ",total_dim)
    model = MLP(total_dim)
    

    if clip_model[0]== "hf-hub:timm":
        model_names="SigLip"
        # Extract model names from clip_models and join them with underscores
    else:
        model_names = "_".join([model[0].replace('/', '').lower() for model in clip_models])

    # Use the combined model names for the logger
    logger = TensorBoardLogger('tb_logs', name=f'my_model_{model_names}',log_graph=True)
    
    
    
    # Create the LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Add the callback to the Trainer
    trainer = pl.Trainer(
    logger=logger,  # This is your TensorBoardLogger
    max_epochs=epochs,
    devices="auto",
    accelerator="auto",
    callbacks=[lr_monitor],
    check_val_every_n_epoch=int((epochs/batch_size)*5)  # Check validation every 10 epochs
    )
    
    
  
    trainer.fit(model, train_loader, val_loader)


    #trainer.save_checkpoint(str(out_path / save_name))

    torch.save(model.state_dict(), out_path /  save_name)


