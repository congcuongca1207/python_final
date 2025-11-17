import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = load_model("garbage_effnetb0.h5")

# Danh sách lớp (chỉnh theo model của bạn)
class_names = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files['image']

    file_path = "static/uploaded.jpg"
    file.save(file_path)

    # Preprocess ảnh
    img = image.load_img(file_path, target_size=(160, 160))   # đúng input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Dự đoán
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    class_name = class_names[class_id]

    return render_template("index.html", prediction=class_name, image_path=file_path)


if __name__ == "__main__":
    app.run(debug=True)
