# Diabetic Retinopathy Classification

A deep learning web application that classifies retinal fundus images into five diabetic retinopathy severity stages using an ImageNet-pretrained Xception model and a Flask interface.

The project combines transfer learning, image preprocessing, model inference, and a simple web workflow with local user authentication and image upload.

> **Note:** This is an educational/experimental prototype and is **not suitable for clinical diagnosis or real-world medical screening**.

---

## Screenshots

| Home & Detection Flow | Authentication (Login / Register) |
| :---: | :---: |
| <img src="screenshots/Home%20Page.png" alt="Home Page" width="380"> | <img src="screenshots/Login%20Page.png" alt="Login Page" width="380"> |
| **Prediction Interface** | **Register Account** |
| <img src="screenshots/Prediction%20Page.png" alt="Prediction Page" width="380"> | <img src="screenshots/Register%20Page.png" alt="Register Page" width="380"> |

---

## How It Works

```text
APTOS 2019 Fundus Images
          │
          ▼
Resize to 299 × 299
          │
          ▼
Xception Preprocessing
+ Training Augmentation
          │
          ▼
ImageNet-pretrained Xception
        (Frozen)
          │
          ▼
Flatten → Dense(256)
→ Dropout(0.5) → Dense(5)
          │
          ▼
5-Class Softmax Prediction
          │
          ▼
Flask Web Application
          │
          ▼
Predicted DR Severity
```

The model predicts one of:

- No DR
- Mild NPDR
- Moderate NPDR
- Severe NPDR
- PDR

---

## Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset.

| Split | Images |
|---|---:|
| Training | 3,662 |
| Testing / Validation | 734 |
| Total | 4,396 |

The dataset contains five severity classes:

| Class | Label |
|---|---|
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | PDR |

The training data is imbalanced. For example, No DR represents about 49.3% of the training set, while Severe NPDR represents about 5.3%.

No class weighting, SMOTE, oversampling, focal loss, or undersampling was used.

> The `testing` directory was used as `validation_data` during training. It was therefore not an independent blind test set.

---

## Model

The project uses **Xception with ImageNet pretrained weights** as the convolutional feature extractor.

The Xception base is completely frozen and a custom classification head is trained on top:

```text
Xception Base
    │
    ▼
10 × 10 × 2048 Feature Map
    │
    ▼
Flatten
    │
    ▼
Dense(256, ReLU)
    │
    ▼
Dropout(0.5)
    │
    ▼
Dense(5, Softmax)
```

### Model Size

- Total parameters: **73.29M**
- Trainable parameters: **52.43M**
- Non-trainable parameters: **20.86M**

The large trainable parameter count comes mainly from flattening the `10 × 10 × 2048` Xception feature map before the dense layer.

---

## Training

Training was performed using TensorFlow/Keras.

### Configuration

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch size: 32
- Maximum epochs: 30
- Actual epochs: 19
- Early stopping patience: 5
- Input size: 299 × 299
- Output classes: 5

### Training Augmentation

The training pipeline applies:

- Rotation: ±20°
- Width shift: 10%
- Height shift: 10%
- Zoom: 20%
- Brightness: 0.8–1.2
- Horizontal flip

The Xception `preprocess_input` function is used during training, scaling image values from `[0, 255]` to `[-1, 1]`.

---

## Results

The best recorded validation result occurred at **Epoch 14**:

| Metric | Result |
|---|---:|
| Validation Accuracy | **73.58%** |
| Validation Loss | **0.73374** |
| Training Accuracy | 75.00% |
| Training Loss | 0.7502 |

Training stopped at Epoch 19 after validation loss stopped improving, and the best weights from Epoch 14 were restored.

### Evaluation Limitation

The project currently evaluates the model primarily using accuracy.

The following were **not implemented**:

- Confusion matrix
- Precision
- Recall
- F1-score
- Specificity
- ROC-AUC
- Quadratic weighted kappa
- Per-class error analysis

Therefore, the 73.58% figure should be treated as a validation result, not as a complete evaluation of clinical or per-class performance.

---

## Web Application

The Flask application provides:

- User registration
- Login and logout
- Password hashing with Werkzeug
- Session-based authentication
- Drag-and-drop image upload
- Image preview before submission
- PNG/JPG/JPEG validation
- Secure filename handling
- UUID-prefixed uploaded filenames
- Model inference
- Predicted severity display
- Reference cards for the five severity stages

The prediction flow is:

```text
User Login
    │
    ▼
Upload Fundus Image
    │
    ▼
Validate Image
    │
    ▼
Convert to RGB
    │
    ▼
Resize to 299 × 299
    │
    ▼
Normalize Image
    │
    ▼
Xception Model
    │
    ▼
Softmax Output
    │
    ▼
Argmax
    │
    ▼
Predicted Severity Label
```

---

## Important Technical Details

### Transfer Learning

The project uses an ImageNet-pretrained Xception model rather than training a CNN from scratch.

The complete Xception feature extractor remains frozen, while the custom classification head is trained.

### Early Stopping

Training monitors validation loss:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

A model checkpoint is also saved whenever validation loss improves.

### Keras 3 Compatibility

The Flask application includes a compatibility helper for older HDF5 model files.

If Keras 3 raises a deserialization error related to `batch_shape`, the application patches the HDF5 model configuration before loading it.

### File Upload Handling

Uploaded filenames are sanitized with `secure_filename()` and prefixed with a UUID to reduce filename collisions.

---

## Project Structure

```text
diabetic-retinopathy-classification/
├── app.py
├── requirements.txt
├── README.md
├── Xception_Diabetic_retinopathy.ipynb
├── users.json
├── model/
│   ├── best_xception_model.h5
│   └── Updated-Xception-diabetic-retinopathy.h5
├── preprocessed dataset/
│   ├── training/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── 2/
│   │   ├── 3/
│   │   └── 4/
│   └── testing/
│       ├── 0/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       └── 4/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── logout.html
│   └── prediction.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
├── uploads/
└── screenshots/
    ├── Home Page.png
    ├── Login Page.png
    ├── Prediction Page.png
    └── Register Page.png
```

> The trained `.h5` model files are approximately 713 MB each locally and may not be included in the GitHub repository because of their size.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashok-shetti/diabetic-retinopathy-classification.git
cd diabetic-retinopathy-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Place the trained model file at:

```text
model/best_xception_model.h5
```

The application loads this model for inference.

### 5. Run the Flask application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Tech Stack

| Area | Technologies |
|---|---|
| Machine Learning | Python, TensorFlow, Keras, Xception, Transfer Learning, ImageDataGenerator, NumPy |
| Backend | Flask, Werkzeug, Gunicorn |
| Computer Vision | Pillow, Image Preprocessing, Image Resizing, Data Augmentation |
| Frontend | HTML5, Jinja2, CSS3, JavaScript |
| Storage | JSON-based Local User Storage, Local Filesystem, HDF5 Model Files |

---

## Current Limitations

This project has several known limitations:

1. **Training/inference preprocessing mismatch**

   Training uses Xception preprocessing with values in `[-1, 1]`, while the Flask inference code currently scales images to `[0, 1]` using `/255.0`.

2. **Class imbalance**

   The model was trained on an imbalanced dataset without class weighting or other imbalance-handling techniques.

3. **Large classification head**

   `Flatten()` produces 204,800 features before the dense layer, resulting in more than 52 million trainable parameters.

4. **Limited evaluation**

   Only validation accuracy and loss were recorded. There is no confusion matrix or per-class precision/recall/F1 analysis.

5. **No independent blind test set**

   The testing directory was used for validation during training, including early stopping and model selection.

6. **No confidence score in the UI**

   The application displays only the predicted class and does not show the softmax probabilities.

7. **File-based authentication**

   User information is stored in `users.json`, which is suitable for a small local prototype but is not designed for concurrent production use.

8. **No explainability**

   The application does not provide Grad-CAM, saliency maps, or other visual explanations for predictions.

9. **No automated ophthalmic preprocessing**

   The project does not implement automated retina cropping, black-border removal, or specialized illumination/color normalization.

---

## Future Improvements

Possible improvements include:

- Fix the training/inference normalization mismatch
- Replace `Flatten()` with `GlobalAveragePooling2D()`
- Add class weighting or focal loss
- Add confusion matrix and per-class metrics
- Create a separate blind test set
- Display prediction confidence
- Add Grad-CAM visualizations
- Add better retinal image preprocessing
- Replace JSON authentication with a database
- Reduce model size and memory requirements
- Evaluate the model using clinically relevant metrics

---

## Disclaimer

This project is an **educational and experimental deep learning prototype**.

It should **not** be used for medical diagnosis, treatment decisions, or real-world clinical screening. The reported 73.58% result is a validation accuracy from the project's training workflow and does not establish clinical performance or reliability.
