from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('model.pkl')

@app.post('/predict')
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return {'prediction': int(prediction[0])}

