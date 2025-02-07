from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Define the path to upload folder
UPLOAD_FOLDER = 'Plant-Dieases-main\\static\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your saved model
MODEL_PATH = 'plant_disease_detection_model_1.h5'  # model's path
model = load_model(MODEL_PATH)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return "File does not exist"

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image as done during training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the label (0: healthy, 1: diseased)
    prediction = model.predict(img_array)
    label = 1 if prediction[0] > 0.6 else 0  # Threshold of 0.5 for binary classification

    # Map label to class name
    class_labels = ['healthy', 'diseased']
    predicted_label = class_labels[label]

    return predicted_label


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Predict the result
            prediction = model_predict(file_path)

            return   render_template('result.html', prediction=prediction.title(), image_path=filename)

    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
