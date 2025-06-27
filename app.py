from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from siamese_diff_model import get_siamese_model

app = Flask(__name__)
HEIGHT, WIDTH = 224, 224
MODEL_PATH = "model.weights.h5"

# Load Siamese model
model = get_siamese_model(HEIGHT)
model.build([(None, HEIGHT, WIDTH, 3), (None, HEIGHT, WIDTH, 3)])
model.load_weights(MODEL_PATH)

# Preprocessing function
def preprocess(image_file):
    img = Image.open(image_file).convert("RGB").resize((WIDTH, HEIGHT))
    img = np.array(img).astype("float32") / 255.0
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image1_file = request.files["image1"]
        image2_file = request.files["image2"]

        # Preprocess images
        img1 = preprocess(image1_file)
        img2 = preprocess(image2_file)

        # Expand dimensions
        img1_exp = np.expand_dims(img1, axis=0)
        img2_exp = np.expand_dims(img2, axis=0)

        # Predict difference map
        prediction = model.predict((img1_exp, img2_exp))[0, :, :, 0]
        binary_map = (prediction > 0.5).astype(np.uint8) * 255
        result_img = Image.fromarray(binary_map)

        # Save input and result images
        os.makedirs("static", exist_ok=True)
        input1_path = os.path.join("static", "input1.png")
        input2_path = os.path.join("static", "input2.png")
        result_path = os.path.join("static", "result.png")

        Image.fromarray((img1 * 255).astype(np.uint8)).save(input1_path)
        Image.fromarray((img2 * 255).astype(np.uint8)).save(input2_path)
        result_img.save(result_path)

        return render_template("index.html",
                               result_image=result_path,
                               input_image1=input1_path,
                               input_image2=input2_path)

    return render_template("index.html",
                           result_image=None,
                           input_image1=None,
                           input_image2=None)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=8000)
