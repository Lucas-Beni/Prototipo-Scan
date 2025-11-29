import os
import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple
from clip_engine import CLIPEngine, get_clip_engine
from utils import get_image_files, load_image_from_path


class ImageIndexer:
    
    def __init__(self, images_folder: str = "images"):
        self.images_folder = images_folder
        self.clip_engine: Optional[CLIPEngine] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_to_filename: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        print("[Indexer] Iniciando indexação...")
        
        self.id_to_filename = {}
        self.embeddings = None
        self.index = None
        
        self.clip_engine = get_clip_engine()
        
        image_files = get_image_files(self.images_folder)
        
        if not image_files:
            print(f"[Indexer] Nenhuma imagem encontrada em '{self.images_folder}'")
            embedding_dim = self.clip_engine.get_embedding_dimension()
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.is_initialized = True
            return True
        
        print(f"[Indexer] Encontradas {len(image_files)} imagens para indexar")
        
        embeddings_list: List[np.ndarray] = []
        
        for idx, (full_path, filename) in enumerate(image_files):
            try:
                print(f"[Indexer] Processando ({idx + 1}/{len(image_files)}): {filename}")
                
                image = load_image_from_path(full_path)
                embedding = self.clip_engine.generate_embedding(image)

                embedding = embedding / np.linalg.norm(embedding)

                embedding = embedding.astype(np.float32)
                
                faiss_idx = len(embeddings_list)
                embeddings_list.append(embedding)
                self.id_to_filename[faiss_idx] = filename
                
            except Exception as e:
                print(f"[Indexer] Erro ao processar {filename}: {e}")
                continue
        
        if not embeddings_list:
            print("[Indexer] Nenhuma imagem foi processada com sucesso")
            embedding_dim = self.clip_engine.get_embedding_dimension()
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.is_initialized = True
            return True
        
        self.embeddings = np.vstack(embeddings_list).astype(np.float32)
        
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"[Indexer] Índice FAISS criado com {self.index.ntotal} vetores")
        self.is_initialized = True
        return True
    
    def search(self, query_image) -> Tuple[Optional[str], float, int]:
        if not self.is_initialized:
            raise RuntimeError("Indexador não foi inicializado. Chame initialize() primeiro.")
        
        if self.index.ntotal == 0:
            return (None, 0.0, 0)
        
        query_embedding = self.clip_engine.generate_embedding(query_image)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, 1)
        
        best_idx = int(indices[0][0])
        raw_similarity = float(distances[0][0])
        
        similarity = max(0.0, min(1.0, raw_similarity))
        percentage = int(round(similarity * 100))
        
        matched_filename = self.id_to_filename.get(best_idx, None)
        
        return (matched_filename, similarity, percentage)
    
    def get_indexed_count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_all_indexed_files(self) -> List[str]:
        return list(self.id_to_filename.values())


_indexer_instance = None


def get_indexer() -> ImageIndexer:
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = ImageIndexer()
    return _indexer_instance
