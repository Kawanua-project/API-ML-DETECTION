from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO



app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'JPEG'])
app.config['MODEL_FILE'] = "model.h5"
app.config['LABELS_FILE'] = "label.txt"

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("model.h5", compile=False)
with open(app.config['LABELS_FILE'], 'r') as file:
    labels = file.read().splitlines()

@app.route("/prediction", methods=["POST"])
def prediction():
    if 'image' not in request.files:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No file part"
            },
            "data": None
        }), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({
            "status": {
                "code": 400,
                "message": "No selected file" 
            },
            "data": None
        }), 400

    if image and allowed_file(image.filename):
        image.seek(0)  
        img = Image.open(BytesIO(image.read())).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32) / 255.0

        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        class_names = labels[index]
        confidence_score = float(prediction[0][index])

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting",
            },
            "data": {
                "endangered_prediction": class_names,
                "confidence": confidence_score
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Client side error"
            },
            "data": None
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
