from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
import jwt  # Make sure to import jwt library
from google.cloud import storage
import os
import MySQLdb.cursors
from dotenv import load_dotenv



load_dotenv()

# Membuat payload untuk token
token_payload = {
    'exp': datetime.utcnow() + timedelta(days=60),
    'iat': datetime.utcnow()
}

# Menyandikan token dengan menggunakan kunci rahasia
encoded_token = jwt.encode(token_payload, os.getenv('JWT_SECRET_KEY'), algorithm='HS256')

print(encoded_token)

app = Flask(__name__)
CORS(app)
app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')
jwt = JWTManager(app)


# database name
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'


mysql = MySQL(app)


app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg', 'jfif'])
app.config['MODEL_FILE'] = "model.h5"
app.config['LABELS_FILE'] = "label.txt"


# Set up Google Cloud Storage
storage_client = storage.Client.from_service_account_json(os.getenv('GCS_KEY_PATH'))
bucket_name = os.getenv('GCS_BUCKET_NAME')

@app.route('/')
def index():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('''SELECT * FROM species''')
    data = cur.fetchall()
    cur.close() 
    return jsonify({
        'status': {
            'code': 200,
            'data': data
        }
    }), 200

    
@app.route('/getfoto')
def indexs():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('''SELECT * FROM upload_photo''')
    data = cur.fetchall()
    cur.close() 
    return jsonify({
        'status': {
            'code': 200,
            'data': data
        }
    }), 200



def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

model = load_model("model.h5", compile=False)
with open(app.config['LABELS_FILE'], 'r') as file:
    labels = file.read().splitlines()

@app.route("/prediction", methods=["POST"])
@jwt_required()
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
        image_content = image.read() 
        img = Image.open(BytesIO(image_content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32) / 255.0

        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        class_names = labels[index]
        confidence_score = float(prediction[0][index])
           
        filename = secure_filename(image.filename)
        blob = storage_client.bucket(bucket_name).blob(filename)
        blob.upload_from_string(image_content, content_type=image.content_type)


        # Dapatkan URL gambar di Cloud Storage
        image_url = f"https://storage.googleapis.com/{bucket_name}/{filename}"


        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute('SELECT kingdom, phylum, class, "order", family, species, nama, deskripsi, tingkat_kelangkaan, habitat FROM species WHERE label = % s', (class_names, ))
        result = cur.fetchone()
        cur.close() 
   
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting",
            },
            "data": {
                "endangered_prediction": class_names,
                "confidence": confidence_score,
                "image_url": image_url, 
                "result" : result
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
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))
