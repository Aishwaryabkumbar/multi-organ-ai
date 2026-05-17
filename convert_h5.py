import tensorflow as tf

# ---------------- LIVER ----------------
model = tf.keras.models.load_model(
    "models/liver_model.keras",
    compile=False
)

model.save("models/liver_model.h5")

print("Liver converted")


# ---------------- KIDNEY ----------------
model = tf.keras.models.load_model(
    "models/kidney_model.keras",
    compile=False
)

model.save("models/kidney_model.h5")

print("Kidney converted")


# ---------------- PANCREAS ----------------
model = tf.keras.models.load_model(
    "models/pancreas_model.keras",
    compile=False
)

model.save("models/pancreas_model.h5")

print("Pancreas converted")


# ---------------- SPLEEN ----------------
model = tf.keras.models.load_model(
    "models/spleen_model.keras",
    compile=False
)

model.save("models/spleen_model.h5")

print("Spleen converted")


# ---------------- GB ----------------
model = tf.keras.models.load_model(
    "models/gb_model.keras",
    compile=False
)

model.save("models/gb_model.h5")

print("GB converted")