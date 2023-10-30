import torch
from PIL import Image
import clip
from tqdm import tqdm
import pandas as pd
import pathlib

from train_predictor import MLP #MLP BertAestheticScorePredictor
from prepare_training_data import normalized
import open_clip


def predict_score(root_folder, database_file, train_from, clip_models):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = database_file.split(".")[0]
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    database = pd.read_csv(database_path)

    #clip_models=[('ViT-B-16', 'openai'),('RN50', 'openai')]



    total_dim = 0
    models = []
    preprocessors = []

    for clip_model in clip_models:
        if clip_model[0]== "hf-hub:timm":
            config=open_clip.get_model_config("ViT-B-16-SigLIP-512")#["embed_dim"]  
        else:
            config = open_clip.get_model_config(clip_model[0])


            
        if config is not None and 'embed_dim' in config:
            total_dim += config['embed_dim']
        else:
            raise ValueError(f"Embedding dimension not found for model {clip_model[0]}")

        if clip_model[0] == "hf-hub:timm":
            model,preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')
            model.to(device)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1], device=device)


        models.append(model)
        preprocessors.append(preprocess)

    # Use the total dimension for the MLP model
    mlp_model = MLP(total_dim)

    
    model_name = f"{prefix}_linear_predictor_concatenated_{train_from}_mse.pth"
    s = torch.load(path / model_name)   # load the model you trained previously or the model available in this repo
    mlp_model.load_state_dict(s)
    mlp_model.to("cuda")
    mlp_model.eval()
    
    
    pred = []

    for i, row in tqdm(database.iterrows(), total=database.shape[0]):
        #Get the image
        pil_image = Image.open(row["path"])

        image_features_list = []

        for j, model in enumerate(models):
            image = preprocessors[j](pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image.to(device)).to(device)
                image_features_list.append(image_features)

        # Concatenate the embeddings along the feature dimension
        concatenated_features = torch.cat(image_features_list, dim=1)

        # Normalize
        im_emb_arr = normalized(concatenated_features.cpu().detach().numpy())

        # Predict
        prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        pred.append(prediction.item())


    if train_from == "label":
        database["label_pred"] = pred
    elif train_from == "score":
        database["score_pred"] = pred

    database.to_csv(database_path,index=False)
    return database


def print_stats(diff):
    print(f"count: {len(diff)}")
    print(f"max: {diff.max():0.4f}")
    print(f"min: {diff.min():0.4f}")
    print(f"mean: {diff.mean():0.4f}")
    print(f"median: {diff.median():0.4f}")
    print(f"var: {diff.var():0.4f}")


def validate_prediction(root_folder,database_file,train_from):

    path = pathlib.Path(root_folder)
    database_path = path / database_file
    df = pd.read_csv(database_path)

    if train_from == "score":

        all = df[df.score!=0]
        diff = all["score"] - all["score_pred"]
        print("all -----------------")
        print_stats(diff)

        for i in range(5,0,-1):
            all = df[df.score==i]
            diff = all["score"] - all["score_pred"]
            print(f"{i} -----------------")
            print_stats(diff)

    if train_from == "label":

        all = df[df.label!=0]
        diff = all["label"] - all["label_pred"]
        print("all -----------------")
        print_stats(diff)

        for i in range(2,0,-1):
            all = df[df.score==i]
            diff = all["label"] - all["label_pred"]
            print(f"{i} -----------------")
            print_stats(diff)

    return