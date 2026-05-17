from flask import Flask, render_template, request, send_file
import os
import numpy as np
import traceback
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

try:
    print("Loading Liver Model...")
    liver_model = tf.keras.models.load_model(
        "models/liver_model.keras",
        compile=False
    )
    print("Liver Loaded")

    print("Loading Kidney Model...")
    kidney_model = tf.keras.models.load_model(
        "models/kidney_model.keras",
        compile=False
    )
    print("Kidney Loaded")

    print("Loading GB Model...")
    gb_model = tf.keras.models.load_model(
        "models/gb_model.keras",
        compile=False
    )
    print("GB Loaded")

    print("Loading Pancreas Model...")
    pancreas_model = tf.keras.models.load_model(
        "models/pancreas_model.keras",
        compile=False
    )
    print("Pancreas Loaded")

except Exception as e:
    print("MODEL LOADING ERROR:")
    traceback.print_exc()

# ---------------- CLASS LABELS ----------------
liver_classes = ["cirrhosis", "fatty_liver", "hcc", "normal"]

kidney_classes = ["Failure", "Normal", "Stone"]

gb_classes = [
    "Cholecystitis",
    "Gallstones",
    "Normal",
    "Polyps_crystals"
]

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

    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, 800, "ABDOMINAL ULTRASOUND REPORT")

    c.setFont("Helvetica", 12)

    c.drawString(50, 760, f"Date: {date}")
    c.drawString(50, 730, f"Organ: {organ}")
    c.drawString(50, 700, f"Disease Prediction: {disease}")

    status = "NORMAL" if disease.lower() == "normal" else "ABNORMAL"

    c.drawString(50, 670, f"Status: {status}")

    c.drawString(50, 620, "AI-Based Automated Report Generation System")

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

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(path)

            img_path = path

            # ---------------- LIVER ----------------
            if organ == "Liver":

                img = prepare_img(path, (224, 224))

                pred = liver_model.predict(img)

                disease = liver_classes[np.argmax(pred)]

            # ---------------- KIDNEY ----------------
            elif organ == "Kidney":

                img = prepare_img(path, (224, 224))

                pred = kidney_model.predict(img)

                disease = kidney_classes[np.argmax(pred)]

            # ---------------- PANCREAS ----------------
            elif organ == "Pancreas":

                img = prepare_img(path, (128, 128))

                recon = pancreas_model.predict(img)

                error = np.mean(np.square(img - recon))

                if error < 0.09584252:
                    disease = "Normal"
                else:
                    disease = "Abnormal"

            # ---------------- SPLEEN ----------------
            elif organ == "Spleen":

                img = prepare_img(path, (224, 224))

                pred = spleen_model.predict(img)

                disease = spleen_classes[np.argmax(pred)]

            # ---------------- GALLBLADDER ----------------
            elif organ == "GB":

                img = prepare_img(path, (224, 224))

                pred = gb_model.predict(img)

                disease = gb_classes[np.argmax(pred)]

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

    pdf_path = generate_pdf(organ, disease)

    return send_file(pdf_path, as_attachment=True)

# ---------------- RUN APP ----------------
# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)