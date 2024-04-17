import argparse
from helpers import initialize_database
from prepare_training_data import prepare_training_data
from train_predictor import train_predictor
from predict_score import predict_score, validate_prediction
from export_prediction import export_prediction
import torch.multiprocessing
import open_clip


# ('ViT-B-16', 'openai'),('ViT-B-32', 'openai')]#,('ViT-L-14', 'openai')]
# P:\python\aesthetics\aestylo\data\Scrapes\3d_test

def main(arguments):
    database_file = "database"
    database = initialize_database(arguments.input, database_file, is_label_from_folder=True)
    clip_model = [("hf-hub:timm", "ViT-SO400M-14-SigLIP-384")]
    prepare_training_data(arguments.input, database_file, "label", clip_model)
    train_predictor(arguments.input, database_file, "label", clip_model)
    predict_score(arguments.input, database_file, "label", clip_model)
    validate_prediction(arguments.input, database_file, "label")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example CLI for processing inputs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input Folder path")
    args = parser.parse_args()
    main(args)
