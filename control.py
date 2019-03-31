from flask import Flask, request, render_template, jsonify
import summarize
import sys
sys.path.append("D:/Fyp/")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/process', methods=['POST'])
def process():
    name = request.form['name']
    if name:
        return jsonify({'name': summarize.summarize(name)})

    return jsonify({'error': 'Missing data!'})


if __name__ == '__main__':
    app.run(debug=True)