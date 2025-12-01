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

trash_info = {
    "battery": "Pin thải là các loại pin đã sử dụng hết hoặc hỏng, bao gồm pin AA, AAA, pin điện thoại, pin laptop, pin đồng hồ, pin điều khiển từ xa, pin sạc dự phòng… Đây là loại rác nguy hại vì chứa nhiều kim loại nặng và hóa chất độc hại như chì, thủy ngân, cadmium, lithium.",
    "biological": "Rác hữu cơ (biological waste) là các loại rác có nguồn gốc từ thực phẩm, động thực vật, có khả năng phân hủy sinh học. Bao gồm thức ăn thừa, rau củ quả hỏng, vỏ trái cây, lá cây, cỏ, thức ăn chăn nuôi, rác từ nhà bếp, xác động vật hoặc thực phẩm thừa. Đây là loại rác phổ biến nhất trong sinh hoạt hằng ngày.",
    "brown-glass": "Brown Glass là các sản phẩm thủy tinh có màu nâu, thường dùng làm chai bia, chai nước ngọt, chai thực phẩm, lọ đựng gia vị. Đây là loại rác có thể tái chế nếu được thu gom và xử lý đúng cách. Thủy tinh nâu giúp chống ánh sáng mặt trời bảo quản sản phẩm, do đó được ưa chuộng trong bao bì thực phẩm và đồ uống.",
    "cardboard": "Cardboard là các sản phẩm giấy cứng dùng làm hộp đựng hàng, bao bì vận chuyển, hộp thực phẩm, vách ngăn, thùng carton. Đây là loại rác có thể tái chế 100% nếu được xử lý đúng cách, đóng vai trò quan trọng trong chuỗi tái chế giấy và bao bì.",
    "clothes": "Rác Clothes là các loại quần áo, giày dép, khăn, vải vụn đã qua sử dụng hoặc hỏng. Đây là loại rác tái sử dụng và tái chế được, nếu được thu gom và xử lý đúng cách, có thể giảm áp lực lên môi trường và giảm lượng rác thải ra bãi rác.",
    "green-glass": "Green Glass là các sản phẩm thủy tinh có màu xanh, thường dùng làm chai rượu vang, chai bia, chai thực phẩm, lọ đựng gia vị. Đây là loại rác có thể tái chế 100% nếu được thu gom và xử lý đúng cách. Thủy tinh xanh giúp bảo vệ sản phẩm khỏi ánh sáng mặt trời, giữ chất lượng đồ uống và thực phẩm.",
    "metal": "Metal là các loại lon nhôm, hộp sắt, nắp chai, đồ gia dụng bằng kim loại đã qua sử dụng hoặc hỏng. Đây là loại rác có thể tái chế 100% nếu được thu gom và xử lý đúng cách. Kim loại tái chế giúp giảm khai thác quặng mới, tiết kiệm năng lượng và bảo vệ môi trường.",
    "paper": "Paper là các loại giấy đã qua sử dụng như giấy báo, giấy in, giấy văn phòng, tập vở cũ, phong bì. Đây là loại rác có thể tái chế 100% nếu được thu gom và xử lý đúng cách. Giấy tái chế giúp giảm khai thác gỗ mới, tiết kiệm năng lượng và bảo vệ môi trường.",
    "plastic": "Plastic là các sản phẩm từ nhựa đã qua sử dụng như chai nhựa, túi nylon, hộp đựng thực phẩm, bao bì nhựa, ly nhựa, đồ chơi, vật dụng gia đình. Đây là loại rác khó phân hủy tự nhiên, thường tồn tại trong môi trường hàng trăm năm, và là nguyên nhân chính gây ra ô nhiễm môi trường nghiêm trọng.",
    "shoes": "Shoes là các loại giày dép, dép, bốt, sandal đã qua sử dụng, hỏng hoặc không còn dùng được. Đây là loại rác có thể tái sử dụng hoặc tái chế một phần, tùy chất liệu của giày (da, vải, cao su, nhựa, kim loại…).",
    "trash": "Trash là rác thải không thể tái chế hoặc khó phân loại, bao gồm các loại rác hỗn hợp như một số loại bao bì hỗn hợp, vỏ đồ ăn nhanh, màng nhựa nhiều lớp, gói thực phẩm trộn, rác bẩn, đồ dùng hư hỏng khó tái chế. Đây là loại rác phổ biến trong sinh hoạt hàng ngày nhưng gây áp lực lớn cho môi trường nếu không được xử lý đúng cách.",
    "white-glass": "White Glass là các sản phẩm thủy tinh trong suốt, thường dùng làm chai nước suối, lọ đựng thực phẩm, ly thủy tinh, lọ mỹ phẩm. Đây là loại rác có thể tái chế 100% nếu được thu gom và xử lý đúng cách. Thủy tinh trắng giúp giữ nguyên màu sắc và chất lượng sản phẩm, đồng thời là nguyên liệu quan trọng trong ngành tái chế thủy tinh."
}



@app.route("/trash_type_list_page")
def trash_type_list_page():
    settings = load_settings()
    return render_template("trash_type_list.html", settings=settings, trash_info=trash_info)
@app.route("/instruction_guide_page")
def instruction_guide_page():
    settings = load_settings()
    return render_template("instruction_guide.html", settings=settings)
@app.route("/trash/<name>")
def trash_detail_page(name):
    settings = load_settings()
    return render_template("trash_detail.html", settings=settings, name=name, info=trash_info[name])

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
            "background_color": "#1a1a2e",
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