##import tensorflow
from flask import Flask, request, render_template
import csv
import math
import os
import shutil
import numpy as np
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'static/file_input'
display_folder = 'static/display'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True

birds = sorted(os.listdir('ttv/train'))
labels = {i: bird for i, bird in enumerate(birds)}


clear_session()
my_model = load_model('model.h5', compile=False)
print('model successfully loaded')

start = [0]
passed = [0]
pack = [[]]
num = [0]

@app.route('/')
def index():
    data = []
    pack = dict()
    with open("csv_files/bird_url.csv", "r") as f:
            rows = csv.reader(f)
            for row in rows:
                bird = dict()
                bird['name'], bird['link'] = row[0], row[1]
                data.append(bird)
                
    return render_template('start.html', birds=data)

@app.route('/uploading')
def blank():
    return render_template('bird_recognize.html', img=None)

@app.route('/upload', methods=['POST'])
def upload():
    #clearing existing files
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))
    file = request.files.getlist("img")
    for f in file:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        num[0] += 1
        print(num)
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)
    pack[0] = []
    return render_template('bird_recognize.html', img=file)


@app.route('/predict')
def predict():
    for i in range(start[0], num[0]):
        info = dict()
        count = 0
        link = ''
        row_no = 0
        pred_name = ""
        data = []
        loc = os.path.join(f'{UPLOAD_FOLDER}/{i+500}.jpg')
        res = []
        img = load_img(loc, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img/255
        img = np.expand_dims(img, [0])
        answer = my_model.predict(img)
        y_class = answer.argmax(axis=-1)
        res = labels[y_class[0]]
        info['prob'] = round(answer.max()*100, 2)
        info['pred'] = res
        with open("csv_files/bird_url.csv", "r") as f:
            rows = csv.reader(f)
            for i, row in enumerate(rows):
                if row[0].lower() == res.lower():
                    link = row[1]
                    count = row[2]
                    row_no = i
                data.append(row)
                    
        with open("csv_files/bird_url.csv", "w", newline="") as file:
            write = csv.writer(file)
            for i, row in enumerate(data):
                if i == row_no:
                    new_count = int(count) + 1
                    row[2] = new_count
                write.writerow(row)
                
        info['img'] = f'{display_folder}/{info["pred"]}.jpg'
        info['url'] = link
        info['count'] = count
        pred_name = res + count + '.jpg'
        
        with open("csv_files/predicted.csv", "a", newline="") as f:
            csv_writer = csv.writer(f)
            save_pred = os.path.join('static', 'predicted', pred_name)
            csv_writer.writerow([res, save_pred])
        
        shutil.copy(loc, os.path.join("static", "predicted", pred_name))
        
        pack[0].append(info)
        print(info)
        print(res)
        print("Url:" + link)
        print(f'Imgpath: {loc}')
        passed[0] += 1

    print("Packed")
    start[0] = passed[0]
    
    return render_template('results.html', pack=pack[0])

@app.route("/history")
def history():
    """history of past records"""
    records = []
    with open("csv_files/predicted.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            records.append(row)
            
    return render_template("history.html", records=records)

if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=12345, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run()
