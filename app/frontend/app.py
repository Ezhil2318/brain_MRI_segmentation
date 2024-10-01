import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose an MRI image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded MRI.', use_column_width=True)

    if st.button('Segment'):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files=files)
        result = response.json()["prediction"]

        pred_mask = np.array(result).astype(np.uint8)
        pred_mask = Image.fromarray(pred_mask.squeeze() * 255)
        st.image(pred_mask, caption='Segmentation Result', use_column_width=True)
