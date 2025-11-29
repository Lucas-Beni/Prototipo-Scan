import requests
import numpy as np
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import time

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

class CLIPEngine:
    def __init__(self):
        if not HF_API_KEY:
            raise ValueError("Você precisa definir a variável de ambiente HF_API_KEY.")
        print(f"[CLIPEngine] Usando HuggingFace API: {HF_URL}")

    def get_embedding_dimension(self):
        return 768

    def _prepare_image(self, image):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return buffered.getvalue()

    def generate_embedding(self, image):
        try:
            img_bytes = self._prepare_image(image)

            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "image/jpeg",
            }

            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(HF_URL, headers=headers, data=img_bytes, timeout=60)

                if response.status_code == 503:
                    try:
                        data = response.json()
                        if "error" in data and "loading" in str(data.get("error", "")).lower():
                            wait_time = data.get("estimated_time", 20)
                            print(f"[CLIPEngine] Modelo carregando, aguardando {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    except:
                        pass
                
                break

            if response.status_code != 200:
                raise RuntimeError(
                    f"Erro HuggingFace {response.status_code}: {response.text[:500]}"
                )

            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'label' in data[0]:
                    labels = sorted(data[:5], key=lambda x: x.get('score', 0), reverse=True)
                    
                    embedding = np.zeros(768, dtype=np.float32)
                    
                    for i, item in enumerate(labels):
                        score = item.get('score', 0)
                        label = item.get('label', '')
                        start_idx = i * 150
                        for j, char in enumerate(label.encode('utf-8')[:150]):
                            if start_idx + j < 768:
                                embedding[start_idx + j] = float(char) / 255.0 * score
                    
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    return embedding
                else:
                    emb = np.array(data, dtype=np.float32).flatten()
                    if len(emb) < 768:
                        emb = np.pad(emb, (0, 768 - len(emb)))
                    elif len(emb) > 768:
                        emb = emb[:768]
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    return emb
            
            raise RuntimeError(f"Formato de resposta inesperado: {type(data)}")

        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding via HuggingFace: {e}")


_engine_instance = None

def get_clip_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CLIPEngine()
    return _engine_instance
