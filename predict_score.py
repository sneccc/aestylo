import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import pathlib
from PIL import Image, UnidentifiedImageError
from train_new import MultiLayerPerceptron
from prepare_training_data import normalized
import open_clip


def predict_score(root_folder, database_file, train_from, clip_models):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = database_file.split(".")[0]
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    database = pd.read_csv(database_path)
    unique_labels = database['label'].unique()
    num_classes = len(unique_labels)
    total_predictions = []
    indices_to_drop = []  # Track indices of failed images

    # clip_models=[('ViT-B-16', 'openai'),('RN50', 'openai')]

    total_dim = 0
    models = []
    preprocessors = []

    for clip_model in clip_models:
        config = open_clip.get_model_config(clip_model[0] if clip_model[0] != "hf-hub:timm" else clip_model[1])
        if 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")

        if clip_model[0] == "hf-hub:timm":
            model, preprocess = open_clip.create_model_from_pretrained(
                clip_model[0] + "/" + clip_model[1])  # for hf-hub:timm/ViT-SO400M-14-SigLIP-384 format
            model.to(device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1],
                                                                         device=device)
        models.append(model)
        preprocessors.append(preprocess)

    # Use the total dimension for the MLP model
    mlp_model = MultiLayerPerceptron(input_size=total_dim,num_classes=num_classes)
    model_name = f"{prefix}_linear_predictor_concatenated_{train_from}_mse.pth"
    mlp_model.load_state_dict(torch.load(path / model_name))
    mlp_model.to(device)
    mlp_model.eval()

    batch_size = 256
    total_predictions = []

    # Process in batches
    for start_idx in tqdm(range(0, len(database), batch_size), desc="Predicting scores"):
        end_idx = start_idx + batch_size
        batch_df = database.iloc[start_idx:end_idx]

        batch_images = []
        current_indices = []  # Track current batch indices for successful processing

        for index, row in batch_df.iterrows():
            try:
                with Image.open(row["path"]) as pil_image:
                    images = [preprocessor(pil_image).unsqueeze(0).to(device) for preprocessor in preprocessors]
                    batch_images.append(torch.cat(images, dim=0))
                    current_indices.append(index)
            except Exception as e:
                print(f"Error on {row['path']},  dropping this index")
                indices_to_drop.append(index)

        if not batch_images:
            continue

        # Further processing
        batch_images_tensor = torch.cat(batch_images, dim=0)
        image_features_list = []
        with torch.no_grad():
            for model in models:
                features = model.encode_image(batch_images_tensor)
                image_features_list.append(features)

        concatenated_features = torch.cat(image_features_list, dim=1)
        im_emb_arr = concatenated_features.cpu().detach().numpy()

        with torch.no_grad(), torch.cuda.amp.autocast():
            prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device))
            predicted_labels = torch.argmax(prediction, dim=1).cpu().numpy()

        total_predictions.extend(predicted_labels.tolist())

    # Drop problematic indices from the original DataFrame
    database = database.drop(indices_to_drop)
    # Ensure total_predictions and database have the same length
    database["label_pred"] = total_predictions
    database.to_csv(database_path, index=False)
    return database

def print_stats(diff):
    print(f"count: {len(diff)}")
    print(f"max: {diff.max():0.4f}")
    print(f"min: {diff.min():0.4f}")
    print(f"mean: {diff.mean():0.4f}")
    print(f"median: {diff.median():0.4f}")
    print(f"var: {diff.var():0.4f}")
def validate_prediction(root_folder, database_file, train_from):
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    df = pd.read_csv(database_path)

    if train_from == "label":

        all = df[df.label != 0]
        diff = all["label"] - all["label_pred"]
        print("all -----------------")
        print_stats(diff)

        for i in range(2, 0, -1):
            all = df[df.score == i]
            diff = all["label"] - all["label_pred"]
            print(f"{i} -----------------")
            print_stats(diff)

    return
