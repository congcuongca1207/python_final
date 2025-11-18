import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("garbage_effnetb0.h5")

class_names = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_page")
def predict_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    path = "static/upload.jpg"
    file.save(path)

    img = image.load_img(path, target_size=(160,160))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)/255.0

    pred = model.predict(arr)
    class_name = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=class_name)

if __name__ == "__main__":
    app.run(debug=True)
