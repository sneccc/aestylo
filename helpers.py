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

def initialize_database_any(root_folder, database_file, is_label_from_folder: bool = False):
    image_path = pathlib.Path(root_folder)
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_path_list = []
    labels = []
    label_names = []
    label_mapping = {}

    if is_label_from_folder:
        # Get all subdirectories and assign labels dynamically
        subdirs = [x for x in image_path.iterdir() if x.is_dir()]
        label_mapping = {subdir.name: idx + 1 for idx, subdir in enumerate(subdirs)}
        for subdir in subdirs:  # Ensure subdir is treated as a Path object
            label = label_mapping[subdir.name]  # Retrieve the label using subdir.name
            for path in subdir.rglob('*'):
                if path.is_file() and path.suffix.lower() in extensions:
                    image_path_list.append(str(path))
                    labels.append(label)
                    label_names.append(subdir.name)  # subdir.name is valid because subdir is a Path object

    # Add images directly under root_folder with default label (0) and name ('Unlabeled')
    for path in image_path.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            # Only include images not in labeled folders
            if all((image_path / subdir.name / path.name).exists() is False for subdir in subdirs):
                image_path_list.append(str(path))
                labels.append(0)
                label_names.append('Unlabeled')

    num = len(image_path_list)
    # Create dataframe
    df = pd.DataFrame({
        'name': [pathlib.Path(path).name for path in image_path_list],
        'path': image_path_list,
        'flag': [0] * num,
        'flag_pred': [0] * num,
        'label': labels,
        'label_name': label_names,
        'label_pred': [0] * num,
        'score': [0] * num,
        'score_pred': [0] * num,
        'show': [True] * num,
    })
    df = df.sort_values(by="name").reset_index(drop=True)

    # Database path
    database_path = pathlib.Path(root_folder) / database_file

    # Create/load database
    if database_path.exists():
        # Load existing database
        old_df = pd.read_csv(database_path)
        # Add new images
        df = pd.concat([old_df, df[~df['name'].isin(old_df['name'])]])
        df = df.sort_values(by="name").reset_index(drop=True)
    df.to_csv(database_path, index=False)

    return df

def get_random_image(database, n_samples=4):
    unmarked_images = database[database['show'] == True]
    random_images = unmarked_images.sample(n_samples)
    return list(random_images['name'])