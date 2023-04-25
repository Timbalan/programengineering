import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app import app


client = TestClient(app)


@pytest.fixture(scope="module")
def test_image():
    # Создание тестового изображения
    image = Image.new(mode='RGB', size=(224, 224), color=(255, 255, 255))
    file_bytes = io.BytesIO()
    image.save(file_bytes, format='JPEG')
    file_bytes.seek(0)
    return file_bytes


def test_predict_success(test_image):
    # Успешный POST запрос на /predict
    response = client.post("/predict", files={"file": test_image})
    assert response.status_code == 200
    assert response.json() == {"result": "tick"}


def test_predict_no_file():
    # Ошибка при POST запросе без передачи файла
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_wrong_file_type(test_image):
    # Ошибка при POST запросе с файлом неправильного формата
    response = client.post("/predict", files={"file": ("test.txt", test_image, "text/plain")})
    assert response.status_code == 422


def test_predict_large_file(test_image):
    # Ошибка при POST запросе с файлом слишком большого размера
    test_image.seek(0, 2)
    size = test_image.tell()
    test_image.seek(0)
    response = client.post("/predict", files={"file": ("test.jpg", test_image, "image/jpeg")})
    assert response.status_code == 413
    assert response.json() == {"detail": f"File too large. Max size is {app.max_file_size} bytes. File size is {size} bytes."}