import os
from typing import Dict, List, Optional, Tuple
from clip_engine import get_image_comparer
from utils import get_image_files, load_image_from_path


class ImageIndexer:
    
    def __init__(self, images_folder: str = "images"):
        self.images_folder = images_folder
        self.image_comparer = None
        self.indexed_images: Dict[str, str] = {}
        self.is_initialized = False
    
    def initialize(self) -> bool:
        print("[Indexer] Iniciando indexação...")
        
        self.indexed_images = {}
        
        self.image_comparer = get_image_comparer()
        
        image_files = get_image_files(self.images_folder)
        
        if not image_files:
            print(f"[Indexer] Nenhuma imagem encontrada em '{self.images_folder}'")
            self.is_initialized = True
            return True
        
        print(f"[Indexer] Encontradas {len(image_files)} imagens para indexar")
        
        for idx, (full_path, filename) in enumerate(image_files):
            try:
                print(f"[Indexer] Indexando ({idx + 1}/{len(image_files)}): {filename}")
                self.indexed_images[filename] = full_path
                
            except Exception as e:
                print(f"[Indexer] Erro ao indexar {filename}: {e}")
                continue
        
        print(f"[Indexer] Índice criado com {len(self.indexed_images)} imagens")
        self.is_initialized = True
        return True
    
    def search(self, query_image) -> Tuple[Optional[str], float, int]:
        if not self.is_initialized:
            raise RuntimeError("Indexador não foi inicializado. Chame initialize() primeiro.")
        
        if len(self.indexed_images) == 0:
            return (None, 0.0, 0)
        
        best_match = None
        best_similarity = 0.0
        
        print(f"[Indexer] Comparando com {len(self.indexed_images)} imagens...")
        
        for filename, full_path in self.indexed_images.items():
            try:
                indexed_image = load_image_from_path(full_path)
                
                similarity = self.image_comparer.compare_images(query_image, indexed_image)
                
                print(f"[Indexer] {filename}: {similarity:.2%}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = filename
                    
            except Exception as e:
                print(f"[Indexer] Erro ao comparar com {filename}: {e}")
                continue
        
        if best_match is None:
            return (None, 0.0, 0)
        
        percentage = int(round(best_similarity * 100))
        
        return (best_match, best_similarity, percentage)
    
    def get_indexed_count(self) -> int:
        return len(self.indexed_images)
    
    def get_all_indexed_files(self) -> List[str]:
        return list(self.indexed_images.keys())


_indexer_instance = None


def get_indexer() -> ImageIndexer:
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = ImageIndexer()
    return _indexer_instance
