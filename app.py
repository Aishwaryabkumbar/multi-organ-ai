from flask import Flask, render_template, request, send_file
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from reportlab.pdfgen import canvas
import datetime
import time

app = Flask(__name__)

# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- MODELS ----------------
liver_model = tf.keras.models.load_model("models/liver_model.keras")
kidney_model = tf.keras.models.load_model("models/kidney_model.keras")
pancreas_model = tf.keras.models.load_model("models/pancreas_model.keras")
spleen_model = tf.keras.models.load_model("models/spleen_model.keras")
gb_model = tf.keras.models.load_model("models/gb_model.keras")

# ---------------- LABELS ----------------
liver_classes = ["cirrhosis", "fatty_liver", "hcc", "normal"]
kidney_classes = ["Failure", "Normal", "Stone"]
gb_classes = ["Cholecystitis", "Gallstones", "Normal", "Polyps_crystals"]
spleen_classes = ["Abnormal", "Normal"]

# ---------------- IMAGE PREPROCESS ----------------
def prepare_img(path, size):
    img = image.load_img(path, target_size=size)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- PDF GENERATION ----------------
def generate_pdf(organ, disease):

    filename = f"{REPORT_FOLDER}/report_{int(time.time())}.pdf"

    c = canvas.Canvas(filename)

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, 800, "ABDOMINAL ULTRASOUND REPORT")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Date: {date}")
    c.drawString(50, 730, f"Organ Analyzed: {organ}")
    c.drawString(50, 710, f"AI Diagnosis: {disease}")

    status = "NORMAL" if disease.lower() == "normal" else "ABNORMAL"

    c.drawString(50, 680, f"Status: {status}")

    c.drawString(50, 640, "AI-Based Medical Decision Support System")

    c.save()

    return filename

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    organ = None
    disease = None
    img_path = None

    if request.method == "POST":

        organ = request.form["organ"]
        file = request.files["image"]

        if file:

            filename = str(int(time.time())) + "_" + file.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            img_path = path

            # ---------------- LIVER ----------------
            if organ == "Liver":
                img = prepare_img(path, (224, 224))
                disease = liver_classes[np.argmax(liver_model.predict(img))]

            # ---------------- KIDNEY ----------------
            elif organ == "Kidney":
                img = prepare_img(path, (224, 224))
                disease = kidney_classes[np.argmax(kidney_model.predict(img))]

            # ---------------- PANCREAS ----------------
            elif organ == "Pancreas":
                img = prepare_img(path, (128, 128))
                recon = pancreas_model.predict(img)
                error = np.mean(np.square(img - recon))
                disease = "Normal" if error < 0.09584252 else "Abnormal"

            # ---------------- SPLEEN ----------------
            elif organ == "Spleen":
                img = prepare_img(path, (224, 224))
                disease = spleen_classes[np.argmax(spleen_model.predict(img))]

            # ---------------- GB ----------------
            elif organ == "GB":
                img = prepare_img(path, (224, 224))
                disease = gb_classes[np.argmax(gb_model.predict(img))]

    return render_template(
        "dashboard.html",
        organ=organ,
        disease=disease,
        img_path=img_path
    )

# ---------------- DOWNLOAD REPORT ----------------
@app.route("/download")
def download():

    organ = request.args.get("organ")
    disease = request.args.get("disease")

    file_path = generate_pdf(organ, disease)

    return send_file(file_path, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)