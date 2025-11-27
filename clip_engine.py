import torch
import clip
from PIL import Image
import numpy as np
from typing import Union


class CLIPEngine:
    """
    Motor CLIP para geração de embeddings de imagens.
    Usa ViT-B/32 por padrão, otimizado para CPU.
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cpu"
        print(f"[CLIPEngine] Carregando modelo {model_name} no dispositivo: {self.device}")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.visual.output_dim
        print(f"[CLIPEngine] Modelo carregado! Dimensão do embedding: {self.embedding_dim}")
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna a dimensão dos vetores de embedding.
        """
        return self.embedding_dim
    
    def generate_embedding(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Gera embedding vetorial para uma imagem.
        
        Args:
            image: Objeto PIL Image ou caminho para arquivo de imagem
            
        Returns:
            Vetor numpy normalizado de embedding
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embedding = image_features.cpu().numpy().astype(np.float32)
        return embedding.squeeze(0)
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula a similaridade de cosseno entre dois embeddings.
        Como os embeddings já são normalizados, o produto escalar é igual à similaridade de cosseno.
        
        Args:
            embedding1: Primeiro vetor de embedding
            embedding2: Segundo vetor de embedding
            
        Returns:
            Valor de similaridade entre 0 e 1
        """
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return float(np.clip(similarity, 0.0, 1.0))


_engine_instance = None


def get_clip_engine() -> CLIPEngine:
    """
    Retorna instância singleton do CLIPEngine.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CLIPEngine()
    return _engine_instance
