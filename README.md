## Face Mask Detector

A small, production‑ready demo that detects whether faces are wearing masks in images or live webcam video. It uses OpenCV for face detection and a MobileNetV2 model (TensorFlow/Keras) for mask classification. A Streamlit UI makes it easy to try.

### What’s inside
- Streamlit web app: image upload, single‑shot webcam, and Live Webcam (auto) modes
- Real‑time predictions with bounding boxes, labels, and confidence
- Training script to retrain the classifier in 5 epochs

### Quick start
Requirements: Python 3.11 recommended.
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
Then open `http://localhost:8501`.

### Dataset (included)
Two classes organized as:
```
dataset/
├─ with_mask/       (~2,165 images)
└─ without_mask/    (~1,930 images)
```

### Train (5 epochs)
```powershell
.\venv\Scripts\python.exe train_model.py --dataset dataset
```
Outputs:
- `mask_detector.model` (Keras H5)
- `plot.png` (training curves)
- Classification report in the console

### How it works
1. OpenCV DNN finds faces in each frame/image.
2. Each face is preprocessed to 224×224 RGB and fed to MobileNetV2.
3. The app overlays Mask/No Mask with confidence.
4. Models are cached to avoid reloading during live inference.

### Tech stack
- TensorFlow/Keras (MobileNetV2), OpenCV DNN, Streamlit, Python 3.11

### Project layout
```
app.py                 # Streamlit UI + inference (includes Live Webcam mode)
train_model.py         # Training script (epochs=5)
mask_detector.model    # Trained model (generated after training)
requirements.txt       # Dependencies
face_detector/         # OpenCV face detector (prototxt + caffemodel)
dataset/               # Training images (with_mask / without_mask)
images/                # Sample images
```

### Notes
- If PowerShell blocks scripts, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- CPU works; GPU is optional.

