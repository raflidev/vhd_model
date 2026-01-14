import os
import io
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model

app = FastAPI()

MODEL_PATH = "model.h5"
CLASSES = ['AS', 'MR', 'MS', 'MVP', 'N']
N_MFCC = 13
MAX_PAD_LEN = 100

# Load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")

def preprocess_audio(file_like):
    try:
        # Load audio using librosa
        # sr=22050 is default for librosa and likely what was used in training as per notebook comments
        audio, sample_rate = librosa.load(file_like, res_type='kaiser_fast', sr=22050)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # Padding or Truncating to ensure fixed shape (n_mfcc, max_pad_len)
        if mfcc.shape[1] > MAX_PAD_LEN:
            mfcc = mfcc[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # Transpose to shape (Time, Features) for LSTM input: (100, 13)
        mfcc = mfcc.T 
        
        # Add batch dimension: (1, 100, 13)
        mfcc = np.expand_dims(mfcc, axis=0)
        
        return mfcc
    except Exception as e:
        print(f"Error parsing audio: {e}")
        return None

@app.get("/")
def read_root():
    model_status = "loaded" if model is not None else "not loaded"
    return {"message": "VHD Audio Classification API", "model_status": model_status}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check file type roughly
    if not file.content_type.startswith("audio/") and not file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        # Just a warning, librosa might handle it anyway
        pass

    try:
        content = await file.read()
        file_like = io.BytesIO(content)
        
        data = preprocess_audio(file_like)
        
        if data is None:
            raise HTTPException(status_code=400, detail="Could not process audio data")
            
        prediction = model.predict(data)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        predicted_class = CLASSES[class_idx]
        
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "class_id": int(class_idx),
            "confidence": confidence,
            "probabilities": {cls: float(prob) for cls, prob in zip(CLASSES, prediction[0])}
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Look for PORT environment variable or default to 8000
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
