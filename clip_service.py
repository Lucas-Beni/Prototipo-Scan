import requests
import numpy as np
from PIL import Image
from io import BytesIO
import os
import time
import base64

HF_API_KEY = os.getenv("HF_API_KEY")

VIT_MODEL = "google/vit-base-patch16-224"
VIT_URL = f"https://router.huggingface.co/hf-inference/models/{VIT_MODEL}"

TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBEDDING_URL = f"https://router.huggingface.co/hf-inference/models/{TEXT_EMBEDDING_MODEL}"

BLIP_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_URL = f"https://router.huggingface.co/hf-inference/models/{BLIP_MODEL}"

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-pt"
TRANSLATION_URL = f"https://router.huggingface.co/hf-inference/models/{TRANSLATION_MODEL}"

IMAGE_EMBEDDING_DIM = 768
TEXT_EMBEDDING_DIM = 384


class CLIPService:
    def __init__(self):
        if not HF_API_KEY:
            raise ValueError("Você precisa definir a variável de ambiente HF_API_KEY.")
        self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        print(f"[CLIPService] Inicializado com modelo ViT: {VIT_MODEL}")

    def get_image_embedding_dimension(self):
        return IMAGE_EMBEDDING_DIM
    
    def get_text_embedding_dimension(self):
        return TEXT_EMBEDDING_DIM

    def _prepare_image(self, image):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return buffered.getvalue()

    def _call_api_with_retry(self, url, data=None, json_data=None, max_retries=3):
        headers = self.headers.copy()
        
        for attempt in range(max_retries):
            try:
                if data is not None:
                    headers["Content-Type"] = "image/jpeg"
                    response = requests.post(url, headers=headers, data=data, timeout=120)
                else:
                    response = requests.post(url, headers=headers, json=json_data, timeout=120)

                if response.status_code == 503:
                    try:
                        error_data = response.json()
                        if "error" in error_data and "loading" in str(error_data.get("error", "")).lower():
                            wait_time = error_data.get("estimated_time", 30)
                            print(f"[CLIPService] Modelo carregando, aguardando {wait_time}s...")
                            time.sleep(min(wait_time, 60))
                            continue
                    except:
                        pass
                    time.sleep(10)
                    continue

                if response.status_code == 429:
                    wait_time = 30
                    print(f"[CLIPService] Rate limit, aguardando {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return response

            except requests.exceptions.Timeout:
                print(f"[CLIPService] Timeout, tentativa {attempt + 1}/{max_retries}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"[CLIPService] Erro na requisição: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise

        return response

    def generate_image_embedding(self, image):
        try:
            img_bytes = self._prepare_image(image)
            
            response = self._call_api_with_retry(VIT_URL, data=img_bytes)

            if response.status_code != 200:
                raise RuntimeError(f"Erro HuggingFace {response.status_code}: {response.text[:500]}")

            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'label' in data[0]:
                    labels = sorted(data[:10], key=lambda x: x.get('score', 0), reverse=True)
                    embedding = np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)
                    
                    for i, item in enumerate(labels):
                        score = item.get('score', 0)
                        label = item.get('label', '')
                        label_bytes = label.encode('utf-8')[:75]
                        start_idx = i * 75
                        for j, char in enumerate(label_bytes):
                            if start_idx + j < IMAGE_EMBEDDING_DIM:
                                embedding[start_idx + j] = float(char) / 255.0 * score
                    
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    return embedding
                else:
                    emb = np.array(data, dtype=np.float32).flatten()
                    if len(emb) < IMAGE_EMBEDDING_DIM:
                        emb = np.pad(emb, (0, IMAGE_EMBEDDING_DIM - len(emb)))
                    elif len(emb) > IMAGE_EMBEDDING_DIM:
                        emb = emb[:IMAGE_EMBEDDING_DIM]
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    return emb

            raise RuntimeError(f"Formato de resposta inesperado: {type(data)}")

        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding de imagem: {e}")

    def generate_text_embedding(self, text):
        try:
            response = self._call_api_with_retry(
                TEXT_EMBEDDING_URL,
                json_data={
                    "inputs": {
                        "source_sentence": text,
                        "sentences": [text]
                    },
                    "options": {"wait_for_model": True}
                }
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    emb = np.array(data, dtype=np.float32).flatten()
                    if len(emb) < TEXT_EMBEDDING_DIM:
                        emb = np.pad(emb, (0, TEXT_EMBEDDING_DIM - len(emb)))
                    elif len(emb) > TEXT_EMBEDDING_DIM:
                        emb = emb[:TEXT_EMBEDDING_DIM]
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    return emb
            
            print(f"[CLIPService] API text embedding falhou ({response.status_code}), usando fallback local")
            return self._generate_text_embedding_fallback(text)

        except Exception as e:
            print(f"[CLIPService] Erro na API: {e}, usando fallback local")
            return self._generate_text_embedding_fallback(text)
    
    def _generate_text_embedding_fallback(self, text):
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**32))
        
        words = text.lower().split()
        embedding = np.zeros(TEXT_EMBEDDING_DIM, dtype=np.float32)
        
        for i, word in enumerate(words[:30]):
            word_bytes = word.encode('utf-8')
            for j, char in enumerate(word_bytes[:12]):
                idx = (i * 12 + j) % TEXT_EMBEDDING_DIM
                embedding[idx] += float(char) / 255.0
        
        embedding += np.random.randn(TEXT_EMBEDDING_DIM).astype(np.float32) * 0.05
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)

    def generate_image_caption(self, image):
        try:
            img_bytes = self._prepare_image(image)
            
            response = self._call_api_with_retry(BLIP_URL, data=img_bytes)

            if response.status_code != 200:
                print(f"[CLIPService] Erro ao gerar caption: {response.status_code}")
                return None

            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    return data[0].get('generated_text', '')
                return str(data[0])
            
            return None

        except Exception as e:
            print(f"[CLIPService] Erro ao gerar caption: {e}")
            return None

    def translate_to_portuguese(self, text):
        try:
            response = self._call_api_with_retry(
                TRANSLATION_URL,
                json_data={"inputs": text}
            )

            if response.status_code != 200:
                print(f"[CLIPService] Erro na tradução: {response.status_code}")
                return text

            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    return data[0].get('translation_text', text)
                return str(data[0])
            
            return text

        except Exception as e:
            print(f"[CLIPService] Erro na tradução: {e}")
            return text

    def generate_explanation(self, query_image, match_image, category_name, category_description, similarity_score):
        try:
            query_caption = self.generate_image_caption(query_image)
            match_caption = self.generate_image_caption(match_image)
            
            if query_caption:
                query_caption_pt = self.translate_to_portuguese(query_caption)
            else:
                query_caption_pt = "imagem de consulta"
            
            if match_caption:
                match_caption_pt = self.translate_to_portuguese(match_caption)
            else:
                match_caption_pt = "imagem correspondente"
            
            similarity_pct = int(similarity_score * 100)
            
            explanation_parts = []
            explanation_parts.append(f"A imagem enviada ({query_caption_pt}) foi comparada com as imagens cadastradas.")
            explanation_parts.append(f"A melhor correspondência encontrada ({match_caption_pt}) pertence à categoria '{category_name}'.")
            
            if category_description:
                explanation_parts.append(f"Esta categoria é definida como: {category_description}.")
            
            if similarity_pct >= 80:
                explanation_parts.append(f"A similaridade visual de {similarity_pct}% indica alta correspondência entre as imagens.")
            elif similarity_pct >= 60:
                explanation_parts.append(f"A similaridade visual de {similarity_pct}% indica correspondência moderada.")
            else:
                explanation_parts.append(f"A similaridade visual de {similarity_pct}% sugere alguma semelhança, mas existem diferenças significativas.")
            
            return " ".join(explanation_parts)

        except Exception as e:
            print(f"[CLIPService] Erro ao gerar explicação: {e}")
            return f"Imagem classificada na categoria '{category_name}' com {int(similarity_score * 100)}% de similaridade."


_service_instance = None


def get_clip_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = CLIPService()
    return _service_instance


def reset_clip_service():
    global _service_instance
    _service_instance = None
