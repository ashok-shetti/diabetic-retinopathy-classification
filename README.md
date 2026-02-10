## Diabetic Retinopathy Detection (Deep Learning)

A Flask web app that loads a pretrained Xception-based model (`.h5`) and predicts diabetic retinopathy severity from uploaded retinal images.

### Features
- User registration and login
- Image upload with validation
- Model inference and label display

### Tech Stack
- Python 3.10+
- Flask
- TensorFlow + Keras
- NumPy, Pillow

### Project Structure
- `app.py` Flask app and model inference
- `model/` pretrained model file
- `templates/` HTML templates
- `static/` CSS/JS assets
- `uploads/` uploaded images

### Setup
1. Create and activate a clean environment.
```powershell
conda create -n dr310 python=3.10 -y
conda activate dr310
```

2. Install dependencies.
```powershell
pip install -r requirements.txt
```

3. Ensure the model file exists.
- Default path: `model/best_xception_model.h5`
- Override with `MODEL_PATH` environment variable if needed.

### Run
```powershell
python app.py
```

App runs at `http://0.0.0.0:5000`.

### Notes
- The model is loaded with Keras and includes a small compatibility patch in `app.py` for legacy `batch_shape` serialization in `.h5` files.
- If you see TensorFlow/Keras compatibility errors, reinstall the versions pinned in `requirements.txt` in a clean environment.

### Configuration
Environment variables:
- `MODEL_PATH`: path to the `.h5` model
- `SECRET_KEY`: Flask secret key

### Labels
The model predicts one of the following classes:
- No DR
- Mild NPDR
- Moderate NPDR
- Severe NPDR
- PDR
