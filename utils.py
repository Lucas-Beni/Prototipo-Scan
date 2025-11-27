import os
from typing import List, Tuple
from PIL import Image
import io


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}


def get_image_files(folder_path: str) -> List[Tuple[str, str]]:
    """
    Retorna lista de tuplas (caminho_completo, nome_do_arquivo) para todas as imagens válidas.
    """
    if not os.path.exists(folder_path):
        return []
    
    image_files = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                image_files.append((full_path, filename))
    
    return sorted(image_files, key=lambda x: x[1])


def load_image_from_path(path: str) -> Image.Image:
    """
    Carrega uma imagem do disco e retorna como objeto PIL Image em RGB.
    """
    image = Image.open(path)
    return image.convert('RGB')


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Carrega uma imagem a partir de bytes (upload) e retorna como objeto PIL Image em RGB.
    """
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert('RGB')


def validate_image_file(file_bytes: bytes) -> bool:
    """
    Valida se os bytes representam uma imagem válida.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        return True
    except Exception:
        return False


def calculate_similarity_percentage(similarity: float) -> int:
    """
    Converte score de similaridade (0-1) para porcentagem (0-100).
    """
    return int(round(similarity * 100))


def normalize_similarity(raw_score: float) -> float:
    """
    Normaliza o score para ficar entre 0 e 1.
    O CLIP usando produto escalar já retorna valores normalizados quando os vetores são normalizados.
    """
    return max(0.0, min(1.0, float(raw_score)))
