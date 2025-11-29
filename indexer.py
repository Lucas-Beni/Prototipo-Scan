import os
import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple
from clip_engine import CLIPEngine, get_clip_engine
from utils import load_image_from_path


class ImageIndexer:
    
    def __init__(self, images_folder: str = "images"):
        self.images_folder = images_folder
        self.clip_engine: Optional[CLIPEngine] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_to_image_id: Dict[int, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.is_initialized = False
    
    def initialize_from_db(self, images: list) -> bool:
        print("[Indexer] Iniciando indexação a partir do banco de dados...")
        
        self.id_to_image_id = {}
        self.embeddings = None
        self.index = None
        
        self.clip_engine = get_clip_engine()
        
        if not images:
            print("[Indexer] Nenhuma imagem encontrada no banco de dados")
            embedding_dim = self.clip_engine.get_embedding_dimension()
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.is_initialized = True
            return True
        
        print(f"[Indexer] Encontradas {len(images)} imagens para indexar")
        
        embeddings_list: List[np.ndarray] = []
        
        for idx, image_record in enumerate(images):
            try:
                full_path = image_record.storage_path
                print(f"[Indexer] Processando ({idx + 1}/{len(images)}): {image_record.filename}")
                
                if not os.path.exists(full_path):
                    print(f"[Indexer] Arquivo não encontrado: {full_path}")
                    continue
                
                image = load_image_from_path(full_path)
                embedding = self.clip_engine.generate_embedding(image)
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.astype(np.float32)
                
                faiss_idx = len(embeddings_list)
                embeddings_list.append(embedding)
                self.id_to_image_id[faiss_idx] = image_record.id
                
            except Exception as e:
                print(f"[Indexer] Erro ao processar {image_record.filename}: {e}")
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
    
    def add_single_image(self, image_record) -> bool:
        if not self.is_initialized or self.clip_engine is None:
            return False
        
        try:
            full_path = image_record.storage_path
            
            if not os.path.exists(full_path):
                print(f"[Indexer] Arquivo não encontrado: {full_path}")
                return False
            
            image = load_image_from_path(full_path)
            embedding = self.clip_engine.generate_embedding(image)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1).astype(np.float32)
            
            faiss.normalize_L2(embedding)
            
            faiss_idx = self.index.ntotal
            self.index.add(embedding)
            self.id_to_image_id[faiss_idx] = image_record.id
            
            print(f"[Indexer] Imagem adicionada ao índice: {image_record.filename}")
            return True
            
        except Exception as e:
            print(f"[Indexer] Erro ao adicionar imagem {image_record.filename}: {e}")
            return False
    
    def search(self, query_image, get_image_by_id_func) -> Tuple[Optional[dict], float, int]:
        if not self.is_initialized:
            raise RuntimeError("Indexador não foi inicializado.")
        
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
        
        image_id = self.id_to_image_id.get(best_idx, None)
        
        if image_id is None:
            return (None, 0.0, 0)
        
        image_record = get_image_by_id_func(image_id)
        
        if image_record is None:
            return (None, 0.0, 0)
        
        return (image_record, similarity, percentage)
    
    def get_indexed_count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_all_indexed_image_ids(self) -> List[int]:
        return list(self.id_to_image_id.values())


_indexer_instance = None


def get_indexer() -> ImageIndexer:
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = ImageIndexer()
    return _indexer_instance


def reset_indexer():
    global _indexer_instance
    _indexer_instance = None
