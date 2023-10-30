import clip
import torch
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import open_clip
from torchvision import transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # ... add more transformations as needed
    # Note: Do not add ToTensor() here because the CLIP preprocess function will handle that
])


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
    

def prepare_training_data(root_folder,database_file,train_from,clip_models):
    #clip_models=[('ViT-B-16', 'openai'),('RN50', 'openai')]

    prefix = database_file.split(".")[0]
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    database = pd.read_csv(database_path)

    if train_from == "label":
        df = database[database.label!=0].reset_index(drop=True)
    elif train_from == "score":
        df = database[database.score!=0].reset_index(drop=True)

    out_path = pathlib.Path(root_folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model, preprocess = clip.load(clip_model, device=device)
    for clip_model in clip_models:
        print("Clip Model -> ",clip_model[0],clip_model[1],"full is ",clip_model," type is ->",type(clip_model))

    models = []
    preprocessors = []

    for clip_model in clip_models:

        if clip_model[0] == "hf-hub:timm":
            model,preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')
            model.to(device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1], device=device)
        
        models.append(model)
        preprocessors.append(preprocess)

    x = []
    y = []
    

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        if train_from == "label":
            average_rating = float(row.label)
        elif train_from == "score":
            average_rating = float(row.score)
        
        if average_rating < 1:
            continue
        
        image_features_list = []
        
        for j, model in enumerate(models):
            try:
                image = preprocessors[j](Image.open(row.path)).unsqueeze(0).to(device)
            except:
                continue

            with torch.no_grad():
                image_features = model.encode_image(image.to(device)).to(device)
                image_features_list.append(image_features)

        # Concatenate the embeddings along the feature dimension
        concatenated_features = torch.cat(image_features_list, dim=1)

        im_emb_arr = concatenated_features.cpu().detach().numpy()
        x.append(normalized(im_emb_arr))
        y.append([average_rating])

    x = np.vstack(x)
    y = np.vstack(y)
    x_out = f"{prefix}_x_concatenated_embeddings.npy"
    y_out = f"{prefix}_y_{train_from}.npy"
    np.save(out_path / x_out, x)
    np.save(out_path / y_out, y)
    return