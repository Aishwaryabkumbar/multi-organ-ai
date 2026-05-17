from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from reportlab.pdfgen import canvas

app = Flask(__name__)

# ---------------- CREATE FOLDERS ----------------
UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    organ = request.form["organ"]

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ---------------- IMAGE PREPROCESS ----------------
    if organ == "Pancreas":
        img_size = (128, 128)
    else:
        img_size = (224, 224)

    img = Image.open(filepath).convert("RGB")
    img = img.resize(img_size)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    disease = ""
    confidence = ""

    # ==================================================
    # LIVER
    # ==================================================
    if organ == "Liver":

        model = tf.keras.models.load_model(
            "models/liver_model.keras",
            compile=False
        )

        prediction = model.predict(img_array)

        classes = [
            "Cirrhosis",
            "Fatty Liver",
            "HCC",
            "Normal"
        ]

        pred_index = np.argmax(prediction)
        disease = classes[pred_index]
        confidence = round(np.max(prediction) * 100, 2)

    # ==================================================
    # KIDNEY
    # ==================================================
    elif organ == "Kidney":

        model = tf.keras.models.load_model(
            "models/kidney_model.keras",
            compile=False
        )

        prediction = model.predict(img_array)

        classes = [
            "Failure",
            "Normal",
            "Stone"
        ]

        pred_index = np.argmax(prediction)
        disease = classes[pred_index]
        confidence = round(np.max(prediction) * 100, 2)

    # ==================================================
    # GALLBLADDER
    # ==================================================
    elif organ == "Gallbladder":

        model = tf.keras.models.load_model(
            "models/gb_model.keras",
            compile=False
        )

        prediction = model.predict(img_array)

        classes = [
            "Cholecystitis",
            "Gallstones",
            "Normal",
            "Polyps_crystals"
        ]

        pred_index = np.argmax(prediction)
        disease = classes[pred_index]
        confidence = round(np.max(prediction) * 100, 2)

    # ==================================================
    # PANCREAS
    # ==================================================
    elif organ == "Pancreas":

        model = tf.keras.models.load_model(
            "models/pancreas_model.keras",
            compile=False
        )

        reconstructed = model.predict(img_array)

        error = np.mean(np.square(img_array - reconstructed))

        threshold = 0.01

        if error < threshold:
            disease = "Normal"
        else:
            disease = "Abnormal"

        confidence = round((1 - error) * 100, 2)

    # ---------------- CREATE REPORT ----------------
    report_path = "report.pdf"

    c = canvas.Canvas(report_path)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, 800, "AI Diagnostic Report")

    c.setFont("Helvetica", 14)
    c.drawString(100, 730, f"Organ: {organ}")

    c.drawString(100, 690, f"Prediction: {disease}")

    c.drawString(100, 650, f"Confidence: {confidence}%")

    c.drawString(100, 610, "Generated using Deep Learning")

    c.save()

    # ---------------- SHOW RESULT ----------------
    return render_template(
        "result.html",
        image_path=filepath,
        organ=organ,
        disease=disease,
        confidence=confidence
    )


# ---------------- DOWNLOAD REPORT ----------------
@app.route("/download")
def download():

    return send_file(
        "report.pdf",
        as_attachment=True
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )