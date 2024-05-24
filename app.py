import pathlib
import pandas as pd
from flask import Flask, render_template, send_file, redirect, request
from helpers import initialize_database
from prepare_training_data import prepare_training_data
from _old.train_predictor import train_predictor
from predict_score import predict_score, validate_prediction
from export_prediction import export_prediction
import torch.multiprocessing
import open_clip
from random import randint
app = Flask(__name__)

@app.route('/')
def images():
    global image_batches, width, current_image_index
    current_image = image_batches[current_image_index] if image_batches else None
    print(f"Current Image: {current_image} with {current_image_index} ID")
    return render_template('index.html', image=current_image,current_image_index=current_image_index, width=width)


@app.route('/start')
def start():
    
    clip_models = [model for model in open_clip.list_pretrained()]
    #print(clip_models)  # Add this line
    return render_template("start.html", clip_models=clip_models)



@app.route('/start_session', methods=['POST'])
def start_session():
    global root_folder, database_file, image_batches, width, database, clip_model, current_image_index
    root_folder = request.form['image-folder']
    database_file = request.form['database-file']
    
    #clip_model = request.form['clip_model']  # Capture the selected clip_model
    # Convert the string representation of the dictionary back to an actual dictionary
    #clip_model = ast.literal_eval(clip_model)
    #print(f"NotUsing_Model: {clip_model}")
    
    # init new session
    is_label_from_folder = 'is_label_from_folder' in request.form
    print("is_label_from_folder: ",is_label_from_folder)
    database = initialize_database(root_folder, database_file,is_label_from_folder)
    new_images = database["name"].tolist()  # Fetch all image names from the database
    image_batches = []
    image_batches.extend(new_images)
    current_image_index = len(image_batches) - 1  # Point to the last image in the batch
    return redirect('/')

@app.route('/refresh')
def refresh():
    global image_batches, current_image_index
    new_images = database["name"].tolist()  # Fetch all image names from the database
    image_batches.extend(new_images)
    # Randomize the current_image_index within the range of available images
    current_image_index = randint(0, len(image_batches) - 1)
    return redirect('/')

@app.route('/back')
def back():
    global current_image_index,image_batches

    if current_image_index > 1:
        print(f"Back index change: {current_image_index} -> {current_image_index-1}")
        current_image_index -= 1
    else:
        print(f"Back index changed to last index")
        current_image_index = len(image_batches) - 1
        
    return redirect('/')

@app.route('/forward')
def forward():
    global current_image_index,image_batches

    if current_image_index < len(image_batches) - 2:
        print(f"Forward index change:{current_image_index} -> {current_image_index+1}")
        current_image_index += 1
    else:
        print(f"Forward index changed to first index")
        current_image_index=0
    
    return redirect('/')



@app.route('/train', methods=['POST'])
def train():
    global train_from
    train_from = request.form['train_from']
    return render_template("training.html")

@app.route('/training')
def training():
    global train_from
    #hf-hub:timm/ViT-SO400M-14-SigLIP-384

    clip_model=[("hf-hub:timm","ViT-SO400M-14-SigLIP-384")]#('ViT-B-16', 'openai'),('ViT-B-32', 'openai')]#,('ViT-L-14', 'openai')]
    prepare_training_data(root_folder,database_file,train_from,clip_model)
    train_predictor(root_folder,database_file,train_from,clip_model)
    predict_score(root_folder,database_file,train_from,clip_model)
    validate_prediction(root_folder,database_file,train_from)
    return redirect('/')

@app.route('/export', methods=['POST'])
def export():
    global export_from
    export_from = request.form['export_from']
    return render_template("exporting.html")

@app.route('/exporting')
def exporting():
    global export_from
    export_prediction(root_folder,database_file,export_from)
    return redirect('/')
    
@app.route('/image/<image_name>')
def image(image_name):
    image_path = database.loc[database["name"]==image_name, "path"].item()
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/assign_metadata/<image_name>', methods=['POST'])
def assign_metadata(image_name):

    global image_batches,root_folder,database_file

    # parse request
    col = request.form['col']
    val = request.form['val']
    show = request.form['show']
    if col in ["score"]:
        val = float(val)
    if col in ["mark"]:
        val = int(val)
    if show == "False":
        show = False
    else:
        show = True

    # update dataframe
    database_path = pathlib.Path(root_folder) / database_file
    image_path = database.loc[database["name"]==image_name, "path"].item()
    database.loc[database["path"]==image_path, col] = val
    database.loc[database["path"]==image_path, "show"] = show
    database.to_csv(database_path, index=False)
    
    return redirect('/forward')

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    # init globals
    clip_model = None
    current_image_index = -1
    root_folder = None
    database_file = None
    n_samples = None
    width = 512
    start_flag = True
    image_batches = []
    image_batch = []
    database = pd.DataFrame({})
    app.debug = True
    app.run()
