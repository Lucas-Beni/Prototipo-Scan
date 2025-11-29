import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

HF_MODEL = "sentence-transformers/clip-ViT-B-32"
HF_URL = f"https://router.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"

class CLIPEngine:
    def __init__(self):
        if not HF_API_KEY:
            raise ValueError("Você precisa definir a variável de ambiente HF_API_KEY.")
        print("[CLIPEngine] Usando HuggingFace API:", HF_URL)

    def get_embedding_dimension(self):
        return 512  # CLIP padrão

    def generate_embedding(self, image):

        try:
            # Converte imagem para base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            encoded_image = base64.b64encode(img_bytes).decode("utf-8")

            payload = {
                "inputs": {
                    "image": encoded_image
                }
            }

            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(HF_URL, headers=headers, json=payload)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Erro HuggingFace {response.status_code}: {response.text}"
                )

            data = response.json()

            emb = np.array(data, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            return emb

        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding via HuggingFace: {e}")


_engine_instance = None

def get_clip_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CLIPEngine()
    return _engine_instance
