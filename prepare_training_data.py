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
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # ... add more transformations as needed
    # Note: Do not add ToTensor() here because the CLIP preprocess function will handle that
])


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def prepare_training_data(root_folder, database_file, train_from, clip_models, labels_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = database_file.split(".")[0]
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    database = pd.read_csv(database_path)

    if train_from == "label":
        df = database[database.label != 0].reset_index(drop=True) #Drop all values with label 0
        df['label'] = df['label'] - 1 #shift the range from [1,2,3] to [0,1,2]

    models = []
    preprocessors = []

    for clip_model in clip_models:
        if clip_model[0] == "hf-hub:timm":
            model, preprocess = open_clip.create_model_from_pretrained(
                clip_model[0] + "/" + clip_model[1],device=device)  # for hf-hub:timm/ViT-SO400M-14-SigLIP-384 format
            model.to(device)
            model.eval()
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1],
                                                                         device=device)
        models.append(model)
        preprocessors.append(preprocess)

    # Prepare to collect data
    x, y = [], []

    # Process images in batches
    batch_size = 128
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]

        # Prepare batch data
        batch_images = []
        ratings = []


        for _, row in batch_df.iterrows():
            rating = 0
            if train_from == "label":
                rating = int(row.label)

            if rating == -1:  # Ignoring -1 labels these are for test other than that should be for training
                print("ignoring label -1")
                continue

            try:
                image = Image.open(row.path).convert('RGB')
                processed_images = [preprocessor(image).unsqueeze(0).to(device) for preprocessor in preprocessors]
                batch_images.append(torch.cat(processed_images, dim=0))
                ratings.append(rating)
                #print("rating -> ", rating)
            except Exception as e:
                print(f"Failed to process image {row.path}: {str(e)}")
                continue

        if not batch_images:
            continue  # Skip if no valid images were processed

        # Stack images and process through models
        batch_images_tensor = torch.cat(batch_images, dim=0)

        image_features_list = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for model in models:
                features = model.encode_image(batch_images_tensor)
                image_features_list.append(features)

        # Concatenate the embeddings along the feature dimension
        concatenated_features = torch.cat(image_features_list, dim=1).cpu().detach().numpy()
        x.extend(concatenated_features)

        ratings = np.array(ratings).astype(int)  # Ensure ratings are integer class labels
        y.extend(ratings)

    # Convert lists to numpy arrays and save
    x = np.vstack(x)
    y = np.array(y).astype(int)
    x_out = f"{prefix}_x_concatenated_embeddings.npy"
    y_out = f"{prefix}_y_{train_from}.npy"
    np.save(path / x_out, x)
    np.save(path / y_out, y)


