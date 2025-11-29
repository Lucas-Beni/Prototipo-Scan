import requests
from PIL import Image
from io import BytesIO
import os

DEEPAI_API_KEY = "quickstart-QUdJIGlzIGNvbWluZy4uLi4K"
DEEPAI_URL = "https://api.deepai.org/api/image-similarity"


class ImageComparer:
    def __init__(self):
        print("[ImageComparer] Usando DeepAI API (gratuita)")

    def _image_to_bytes(self, image):
        buffered = BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        buffered.seek(0)
        return buffered.getvalue()

    def compare_images(self, image1, image2):
        try:
            img1_bytes = self._image_to_bytes(image1)
            img2_bytes = self._image_to_bytes(image2)
            
            files = {
                'image1': ('image1.jpg', img1_bytes, 'image/jpeg'),
                'image2': ('image2.jpg', img2_bytes, 'image/jpeg')
            }
            
            headers = {
                'api-key': DEEPAI_API_KEY
            }
            
            response = requests.post(DEEPAI_URL, files=files, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"[ImageComparer] Erro DeepAI {response.status_code}: {response.text}")
                return 0.0
            
            data = response.json()
            
            if 'output' in data and 'distance' in data['output']:
                distance = float(data['output']['distance'])
                similarity = max(0, 100 - distance) / 100
                return similarity
            elif 'output' in data:
                distance = float(data['output'])
                similarity = max(0, 100 - distance) / 100
                return similarity
            
            print(f"[ImageComparer] Resposta inesperada: {data}")
            return 0.0
            
        except Exception as e:
            print(f"[ImageComparer] Erro ao comparar imagens: {e}")
            return 0.0


_comparer_instance = None


def get_image_comparer():
    global _comparer_instance
    if _comparer_instance is None:
        _comparer_instance = ImageComparer()
    return _comparer_instance
