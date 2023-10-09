import pathlib
import pandas as pd
from typing import List

def initialize_database(root_folder, database_file, is_label_from_folder: bool = False):
    
    image_path = pathlib.Path(root_folder)
    extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_path_list = []
    labels = []

    if is_label_from_folder:
        # If labels should be derived from folder names
        good_folder = image_path / "good"
        bad_folder = image_path / "bad"

        for path in good_folder.rglob('*'):
            if path.is_file() and path.suffix in extensions:
                image_path_list.append(str(path))
                labels.append(2)  # Label for good

        for path in bad_folder.rglob('*'):
            if path.is_file() and path.suffix in extensions:
                image_path_list.append(str(path))
                labels.append(1)  # Label for bad

    # Add images directly under root_folder with default label
    for path in image_path.iterdir():
        if path.is_file() and path.suffix in extensions:
            # If the image is also inside /good or /bad, remove it from root_folder
            if (image_path / "good" / path.name).exists() or (image_path / "bad" / path.name).exists():
                path.unlink()
            else:
                image_path_list.append(str(path))
                labels.append(0)  # Default label

    num = len(image_path_list)

    # create dataframe
    df = pd.DataFrame({'name': [pathlib.Path(path).name for path in image_path_list],
                       'path': image_path_list,
                       'flag': [0] * num,
                       'flag_pred': [0] * num,
                       'label': labels,
                       'label_pred': [0] * num,
                       'score': [0] * num,
                       'score_pred': [0] * num,
                       'show': [True] * num,})
    df = df.sort_values(by="name").reset_index(drop=True)

    # database path
    database_path = pathlib.Path(root_folder) / database_file

    # create/load database
    if database_path.exists():
        # load existing database
        old_df = pd.read_csv(database_path)
        # add new images
        df = pd.concat([old_df, df[~df['name'].isin(old_df['name'])]])
        df = df.sort_values(by="name").reset_index(drop=True)
        df.to_csv(database_path, index=False)
    else:
        # create new database
        df.to_csv(database_path, index=False)

    return df


def get_random_image(database, n_samples=4):
    unmarked_images = database[database['show'] == True]
    random_images = unmarked_images.sample(n_samples)
    return list(random_images['name'])