from flask import Flask, request, render_template, send_file, url_for
import os
from inference import run_segmentation
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
OUTPUT_PATH = os.path.join(STATIC_FOLDER, 'output.png')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    run_segmentation(filepath, OUTPUT_PATH)

    # ✅ Return image file as response for frontend JS to show
    return send_file(OUTPUT_PATH, mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True)
