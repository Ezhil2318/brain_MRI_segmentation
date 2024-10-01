from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load the best model
model = tf.keras.models.load_model('../../checkpoints/best_model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert('L')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch and channel dims
    img_array = np.expand_dims(img_array, axis=-1)

    # Get prediction
    pred_mask = model.predict(img_array)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    return {"prediction": pred_mask.tolist()}
