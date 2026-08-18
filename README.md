# Diabetic Retinopathy Classification

A Deep Learning application for classifying diabetic retinopathy severity from retinal images using an ImageNet-pretrained Xception model and a Flask web interface.

## Features

- User registration and login
- Retinal image upload with validation
- Five-class diabetic retinopathy classification
- Xception-based model inference
- Predicted severity label display

## Authentication

The application includes a basic user registration and login system.

- User details are stored in `users.json`.
- Passwords are stored as Werkzeug-generated password hashes rather than plain text.
- Login credentials are validated using the stored password hashes.
- Flask sessions are used to maintain authenticated user sessions.

> **Note:** Authentication data is stored locally in a JSON file; no database is used for user authentication.


## Tech Stack

- **Language:** Python 3.10+
- **Web Framework:** Flask
- **Deep Learning:** TensorFlow, Keras, Xception
- **Image Processing:** NumPy, Pillow
- **Model Format:** H5

## Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset, originally published on Kaggle.

The dataset contains retinal images classified into five diabetic retinopathy severity levels:

- **No DR**
- **Mild NPDR**
- **Moderate NPDR**
- **Severe NPDR**
- **PDR**

### Dataset Split

| Split | Images | Classes |
|---|---:|---:|
| Training | 3,662 | 5 |
| Testing | 734 | 5 |

## Model Architecture

The project uses **Xception** with pretrained **ImageNet weights** as the feature extraction base.

The Xception base layers are frozen during training, and only the custom classification head is trained for the five diabetic retinopathy severity classes.

```text
Input Image (299 × 299)
        │
        ▼
Xception (ImageNet Pretrained)
        │
        │ Frozen Base
        ▼
Flatten
        │
        ▼
Dense (256, ReLU)
        │
        ▼
Dropout (0.5)
        │
        ▼
Dense (5, Softmax)
        │
        ▼
DR Severity Class
```

### Classification Classes

The final softmax layer predicts one of five classes:

1. No DR
2. Mild NPDR
3. Moderate NPDR
4. Severe NPDR
5. PDR

## Training Configuration

The Xception base model was kept frozen while the custom classification head was trained using the following configuration:

| Parameter | Value |
|---|---|
| Image Size | 299 × 299 |
| Batch Size | 32 |
| Maximum Epochs | 30 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |
| Early Stopping | Patience = 5 |

Training stopped early at **Epoch 19**, with the best validation accuracy of **73.58%** achieved at **Epoch 14**. The best model weights were restored after early stopping.

## Results

The model achieved the following performance during training:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **73.58%** |
| Training Accuracy at Best Epoch | **75.00%** |
| Best Epoch | **14** |

Training continued until **Epoch 19**, when the EarlyStopping callback terminated training. The weights from the best-performing epoch were restored.

> **Note:** Precision, Recall, F1 Score, and a confusion matrix were not calculated as part of the original training workflow.

## Project Structure

```text
diabetic-retinopathy-classification/
├── app.py                 # Flask application and model inference
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
├── static/                # CSS and JavaScript assets
└── README.md              # Project documentation
```
### Local / Runtime Files

The following files and directories are used locally or generated at runtime and are excluded from Git:

- `model/best_xception_model.h5` — trained Xception model (~680 MB)
- `users.json` — locally stored user account data and password hashes
- `uploads/` — runtime directory for uploaded retinal images

## Setup

1. Create and activate a clean environment.

```powershell
conda create -n dr310 python=3.10 -y
conda activate dr310
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Add the trained model.

The trained Xception model is not included in this repository because the `.h5` model file is approximately **680 MB**.

Place the model at:

```text
model/best_xception_model.h5
```

The application uses this path by default. To use a different model location, set the `MODEL_PATH` environment variable.

## Run
```powershell
python app.py
```

App runs locally at `http://127.0.0.1:5000/`.

## Screenshots

The application includes the following interface screens:

- **Registration & Login:** User authentication interface.
- **Image Upload:** Interface for uploading retinal images for prediction.
- **Prediction Result:** Displays the predicted diabetic retinopathy severity class.

## Configuration

The application supports the following environment variables:

- `MODEL_PATH`: Optional path to the trained `.h5` model. Defaults to `model/best_xception_model.h5`.
- `SECRET_KEY`: Secret key used by Flask for session management.

## Compatibility Notes

The application includes a compatibility patch in `app.py` for legacy `batch_shape` serialization in the `.h5` model.

If TensorFlow/Keras compatibility errors occur, use the versions specified in `requirements.txt` within a clean Python environment.

