import os

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from starlette.responses import HTMLResponse
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import io
import uvicorn

app = FastAPI()

# загрузка предварительно обученной модели ResNet50 из TensorFlow
model = tf.keras.applications.ResNet50(weights='imagenet')

@app.get("/")
async def main():
    with open("main.html") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

@app.post("/predict")
async def predict(file: UploadFile):
    # чтение загруженного изображения и преобразование его в формат PIL.Image
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB')

    # изменение размера изображения на 224x224 и применение предварительной обработки, используемой моделью ResNet50
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = tf.expand_dims(image_array, 0)

    # предсказание класса заболевания на основе входного изображения
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    class_name = decoded_predictions[0][1]

    # возвращение результата предсказания в формате JSON
    return {"result": class_name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))