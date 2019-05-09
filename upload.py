import os
from flask import Flask, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import intg
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/fyp")
db=client.quesgen

UPLOAD_FOLDER = '/Users/apple/personal/fyp/input_data/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    with open(app.config['UPLOAD_FOLDER']+filename, "r") as f:
        text = f.read()
        question = intg.summarize(text)
        q = {"question": question}
        db.questions.insert_one(q)
    return jsonify({'name': question})


@app.route('/process', methods=['POST'])
def process():
    name = request.form['name']
    if name:
        question = intg.summarize(name)
        q = {"question": question}
        db.questions.insert_one(q)
        return jsonify({'name': question})

    return jsonify({'error': 'Missing data!'})


if __name__ == '__main__':
    app.run(debug=True)
