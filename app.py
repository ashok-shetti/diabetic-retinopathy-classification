from __future__ import annotations

import json
import os
import tempfile
import uuid
from functools import wraps
from typing import Dict

import numpy as np
import tensorflow as tf
import keras
import h5py
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ================= CONFIG =================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "model", "best_xception_model.h5"),
)
USERS_FILE = os.path.join(BASE_DIR, "users.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

LABELS = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= APP =================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ================= MODEL LOAD =================

def _patch_batch_shape_h5(src_path: str) -> str:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    with open(src_path, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(src.read())
    with h5py.File(tmp_path, "r+") as f:
        model_config = f.attrs.get("model_config")
        if model_config is None:
            return tmp_path
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        if "batch_shape" not in model_config:
            return tmp_path
        model_config = model_config.replace('"batch_shape"', '"batch_input_shape"')
        f.attrs.modify("model_config", model_config.encode("utf-8"))
    return tmp_path

# Using keras.models.load_model directly to handle Keras 3 serialization
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except TypeError as e:
    if "batch_shape" not in str(e):
        raise
    print("Detected legacy batch_shape in model config. Applying compatibility patch.")
    patched_path = _patch_batch_shape_h5(MODEL_PATH)
    model = keras.models.load_model(patched_path, compile=False)
    print("Model loaded successfully after patching.")

# ================= USER MANAGEMENT =================

def load_users() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users: Dict[str, Dict[str, str]]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

# ================= AUTH =================

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_email"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

# ================= ML PIPELINE =================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((299, 299)) # Xception default input size

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_label(image_path: str) -> str:
    arr = preprocess_image(image_path)
    
    # Keras 3 models still work with .predict()
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])

    return LABELS[idx]

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    pred = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            pred = "All fields are required."
        else:
            users = load_users()
            if email in users:
                pred = "Account already exists."
            else:
                users[email] = {
                    "name": name,
                    "password_hash": generate_password_hash(password),
                }
                save_users(users)
                session["user_email"] = email
                session["user_name"] = name
                return redirect(url_for("prediction"))

    return render_template("register.html", pred=pred)

@app.route("/login", methods=["GET", "POST"])
def login():
    pred = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        users = load_users()
        user = users.get(email)

        if not user or not check_password_hash(user.get("password_hash", ""), password):
            pred = "Invalid email or password."
        else:
            session["user_email"] = email
            session["user_name"] = user.get("name", "")
            return redirect(url_for("prediction"))

    return render_template("login.html", pred=pred)

@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")

@app.route("/prediction")
@login_required
def prediction():
    return render_template("prediction.html", prediction="", image_url="")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return render_template("prediction.html", prediction="No file uploaded.", image_url="")

    file = request.files["file"]
    if file.filename == "":
        return render_template("prediction.html", prediction="No file selected.", image_url="")

    if not allowed_file(file.filename):
        return render_template("prediction.html", prediction="Unsupported file type.", image_url="")

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    prediction_label = predict_label(file_path)
    image_url = url_for("uploaded_file", filename=filename)

    return render_template(
        "prediction.html",
        prediction=prediction_label,
        image_url=image_url,
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
