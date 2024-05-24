import pandas as pd
import shutil
from pathlib import Path


def export_prediction(root_folder, database_file, export_from, labels_dict):
    assert labels_dict
    prefix = database_file.split(".")[0]
    database_path = Path(root_folder) / database_file
    df = pd.read_csv(database_path)

    if export_from == "label":
        df = df[["path", "label"]]
        export_folder = Path(root_folder) / f"{prefix}_export_label"

        # Create subfolders dynamically based on labels_dict
        subfolders = {}
        for label, subfolder_name in labels_dict.items():
            subfolder_path = export_folder / subfolder_name
            subfolder_path.mkdir(parents=True, exist_ok=True)
            subfolders[label] = subfolder_path

        for index, row in df.iterrows():
            image_path = Path(root_folder) / row["path"]
            if row["label"] in subfolders:
                shutil.copy2(image_path, subfolders[row["label"]])

    elif export_from == "label_pred":
        df = df[["path", "label_name", "label_pred"]]
        export_folder = Path(root_folder) / f"{prefix}_export_label_pred"

        # Create subfolders dynamically based on labels_dict
        subfolders = {}
        for label, subfolder_name in labels_dict.items():
            subfolder_path = export_folder / subfolder_name
            subfolder_path.mkdir(parents=True, exist_ok=True)
            subfolders[label] = subfolder_path

        skipped = 0

        for index, row in df.iterrows():

            if not row["label_name"] == "Unlabeled":
                continue

            image_path = Path(root_folder) / row["path"]
            filename = image_path.stem
            ext = image_path.suffix
            new_filename = f"{filename}_Score_{float(row['label_pred']):.2f}{ext}"

            if row["label_pred"] in subfolders:
                shutil.copy2(image_path, subfolders[row["label_pred"]] / new_filename)
            else:
                skipped += 1

        # Print min, max, and average score and skipped
        print("Minimum score:", df["label_pred"].min())
        print("Maximum score:", df["label_pred"].max())
        print("Average score:", df["label_pred"].mean())
        print("Total images skipped:", skipped)

    return
