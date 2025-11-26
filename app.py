import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Tăng lên 16MB

model = load_model("garbage_effnetb0.h5")

class_names = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

# ==========================
# SETTINGS
# ==========================

SETTINGS_FILE = "settings.json"


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        default_settings = {
            "theme": "dark",
            "input_size": 160,
            "confidence_threshold": 0.5
        }
        save_settings(default_settings)
        return default_settings

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    settings = load_settings() 
    return render_template("home.html", settings=settings) 

@app.route("/predict_page")
def predict_page():
    settings = load_settings()
    return render_template("index.html", settings=settings)

# ==========================
# PREDICT (UPLOAD IMAGE)
# ==========================

@app.route("/predict", methods=["POST"])
def predict():
    settings = load_settings()
    input_size = settings["input_size"]

    file = request.files["image"]
    path = "static/upload.jpg"
    file.save(path)

    img = image.load_img(path, target_size=(input_size, input_size))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    pred = model.predict(arr)

    class_name = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    if confidence < settings["confidence_threshold"]:
        class_name = "Unknown / Low confidence"

    return render_template("index.html",
                           prediction=class_name,
                           confidence=round(float(confidence), 3),
                           settings=settings)


# ==========================
# PREDICT FROM CAMERA
# ==========================

@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    settings = load_settings()
    input_size = settings["input_size"]

    data = request.form.get("imageBase64")

    if not data:
        return "No image received", 400

    try:
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print("Decode error:", e)
        return "Image decode failed", 400

    img = img.resize((input_size, input_size))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    pred = model.predict(arr)
    class_name = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    if confidence < settings["confidence_threshold"]:
        class_name = "Unknown / Low confidence"

    return render_template("index.html",
                           prediction=class_name,
                           confidence=round(float(confidence), 3),
                           settings=settings)

# ==========================
# SETTINGS
# ==========================
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        default_settings = {
            "theme": "dark",
            "input_size": 160,
            "confidence_threshold": 0.5,
            "background_color": "#1a1a2e"  
        }
        save_settings(default_settings)
        return default_settings

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)
@app.route("/settings", methods=["GET", "POST"])
def settings():
    settings = load_settings()

    if request.method == "POST":
        settings["theme"] = request.form.get("theme")
        settings["input_size"] = int(request.form.get("input_size"))
        settings["confidence_threshold"] = float(request.form.get("confidence_threshold"))
        settings["background_color"] = request.form.get("background_color") or "#1a1a2e"
        save_settings(settings)

        return render_template("settings.html", settings=settings, message="Saved successfully!")

    return render_template("settings.html", settings=settings)



@app.route("/info")
def info():
    settings = load_settings()
    return render_template("info.html", settings=settings)  # Tạo file info.html nếu cần

@app.route("/help")
def help():
    settings = load_settings()
    return render_template("help.html", settings=settings)  # Tạo file help.html nếu cần

@app.route("/realtime")
def realtime():
    settings = load_settings()
    return render_template("realtime.html", settings=settings)  # Tạo file realtime.html nếu cần

if __name__ == "__main__":
    app.run(debug=True)