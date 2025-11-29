import os
import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple
from clip_service import CLIPService, get_clip_service
from utils import load_image_from_path


class ImageIndexer:
    
    def __init__(self, images_folder: str = "images"):
        self.images_folder = images_folder
        self.clip_service: Optional[CLIPService] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_to_image_id: Dict[int, int] = {}
        self.image_id_to_category_id: Dict[int, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.category_embeddings: Dict[int, np.ndarray] = {}
        self.is_initialized = False
    
    def initialize_from_db(self, images: list, categories: list = None) -> bool:
        print("[Indexer] Iniciando indexação a partir do banco de dados...")
        
        self.id_to_image_id = {}
        self.image_id_to_category_id = {}
        self.embeddings = None
        self.index = None
        self.category_embeddings = {}
        
        self.clip_service = get_clip_service()
        
        if categories:
            self._initialize_category_embeddings(categories)
        
        if not images:
            print("[Indexer] Nenhuma imagem encontrada no banco de dados")
            embedding_dim = self.clip_service.get_image_embedding_dimension()
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
                embedding = self.clip_service.generate_image_embedding(image)
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.astype(np.float32)
                
                faiss_idx = len(embeddings_list)
                embeddings_list.append(embedding)
                self.id_to_image_id[faiss_idx] = image_record.id
                self.image_id_to_category_id[image_record.id] = image_record.category_id
                
            except Exception as e:
                print(f"[Indexer] Erro ao processar {image_record.filename}: {e}")
                continue
        
        if not embeddings_list:
            print("[Indexer] Nenhuma imagem foi processada com sucesso")
            embedding_dim = self.clip_service.get_image_embedding_dimension()
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
    
    def _initialize_category_embeddings(self, categories):
        print("[Indexer] Gerando embeddings das categorias com modelo multilíngue...")
        for category in categories:
            if category.description:
                try:
                    embedding = self.clip_service.generate_text_embedding(category.description)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    self.category_embeddings[category.id] = embedding
                    category.set_embedding(embedding)
                    print(f"[Indexer] Embedding regenerado para categoria: {category.name} (descrição: {category.description})")
                except Exception as e:
                    print(f"[Indexer] Erro ao gerar embedding para categoria {category.name}: {e}")
    
    def update_category_embedding(self, category):
        if not self.clip_service:
            self.clip_service = get_clip_service()
        
        if category.description:
            try:
                embedding = self.clip_service.generate_text_embedding(category.description)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                self.category_embeddings[category.id] = embedding
                category.set_embedding(embedding)
                print(f"[Indexer] Embedding atualizado para categoria: {category.name}")
                return True
            except Exception as e:
                print(f"[Indexer] Erro ao atualizar embedding da categoria {category.name}: {e}")
                return False
        else:
            if category.id in self.category_embeddings:
                del self.category_embeddings[category.id]
            category.set_embedding(None)
        return True
    
    def add_single_image(self, image_record) -> bool:
        if not self.is_initialized or self.clip_service is None:
            return False
        
        try:
            full_path = image_record.storage_path
            
            if not os.path.exists(full_path):
                print(f"[Indexer] Arquivo não encontrado: {full_path}")
                return False
            
            image = load_image_from_path(full_path)
            embedding = self.clip_service.generate_image_embedding(image)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1).astype(np.float32)
            
            faiss.normalize_L2(embedding)
            
            faiss_idx = self.index.ntotal
            self.index.add(embedding)
            self.id_to_image_id[faiss_idx] = image_record.id
            self.image_id_to_category_id[image_record.id] = image_record.category_id
            
            print(f"[Indexer] Imagem adicionada ao índice: {image_record.filename}")
            return True
            
        except Exception as e:
            print(f"[Indexer] Erro ao adicionar imagem {image_record.filename}: {e}")
            return False
    
    def search(self, query_image, get_image_by_id_func, top_k: int = 5, category_weight: float = 0.3) -> Tuple[Optional[dict], float, int, str, List[dict]]:
        if not self.is_initialized:
            raise RuntimeError("Indexador não foi inicializado.")
        
        if self.index.ntotal == 0:
            return (None, 0.0, 0, "Nenhuma imagem no índice", [])
        
        self._query_text_embedding = None
        
        query_embedding = self.clip_service.generate_image_embedding(query_image)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        candidates = []
        for i in range(k):
            faiss_idx = int(indices[0][i])
            image_similarity = float(distances[0][i])
            image_similarity = max(0.0, min(1.0, image_similarity))
            
            image_id = self.id_to_image_id.get(faiss_idx, None)
            if image_id is None:
                continue
            
            image_record = get_image_by_id_func(image_id)
            if image_record is None:
                continue
            
            category_similarity = 0.0
            if image_record.category_id in self.category_embeddings:
                cat_embedding = self.category_embeddings[image_record.category_id]
                if not hasattr(self, '_query_text_embedding') or self._query_text_embedding is None:
                    query_caption = self.clip_service.generate_image_caption(query_image)
                    if query_caption:
                        query_caption_pt = self.clip_service.translate_to_portuguese(query_caption)
                        print(f"[Indexer] Caption traduzido: {query_caption} -> {query_caption_pt}")
                        self._query_text_embedding = self.clip_service.generate_text_embedding(query_caption_pt)
                        norm = np.linalg.norm(self._query_text_embedding)
                        if norm > 0:
                            self._query_text_embedding = self._query_text_embedding / norm
                    else:
                        self._query_text_embedding = None
                
                if self._query_text_embedding is not None:
                    cat_norm = np.linalg.norm(cat_embedding)
                    if cat_norm > 0:
                        cat_embedding_normalized = cat_embedding / cat_norm
                    else:
                        cat_embedding_normalized = cat_embedding
                    category_similarity = float(np.dot(self._query_text_embedding.flatten(), cat_embedding_normalized.flatten()))
                    category_similarity = max(0.0, min(1.0, category_similarity))
            
            combined_score = (1 - category_weight) * image_similarity + category_weight * category_similarity
            
            candidates.append({
                'image_record': image_record,
                'image_similarity': image_similarity,
                'category_similarity': category_similarity,
                'combined_score': combined_score
            })
        
        if not candidates:
            return (None, 0.0, 0, "Nenhuma correspondência encontrada", [])
        
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        best = candidates[0]
        best_image = best['image_record']
        best_similarity = best['combined_score']
        percentage = int(round(best_similarity * 100))
        
        explanation = self._generate_explanation(
            query_image, 
            best_image, 
            best['image_similarity'],
            best['category_similarity']
        )
        
        alternatives = []
        for i, cand in enumerate(candidates[1:4]):
            alternatives.append({
                'image_id': cand['image_record'].id,
                'filename': cand['image_record'].filename,
                'category_name': cand['image_record'].category.name if cand['image_record'].category else "Sem categoria",
                'combined_score': round(cand['combined_score'] * 100, 1),
                'image_similarity': round(cand['image_similarity'] * 100, 1),
                'category_similarity': round(cand['category_similarity'] * 100, 1)
            })
        
        return (best_image, best_similarity, percentage, explanation, alternatives)
    
    def _generate_explanation(self, query_image, best_image, image_similarity, category_similarity):
        try:
            match_image = load_image_from_path(best_image.storage_path)
            
            category_name = best_image.category.name if best_image.category else "Sem categoria"
            category_description = best_image.category.description if best_image.category else None
            
            combined_similarity = (image_similarity * 0.7) + (category_similarity * 0.3)
            
            explanation = self.clip_service.generate_explanation(
                query_image,
                match_image,
                category_name,
                category_description,
                combined_similarity,
                image_similarity=image_similarity,
                category_similarity=category_similarity
            )
            
            return explanation
            
        except Exception as e:
            print(f"[Indexer] Erro ao gerar explicação: {e}")
            if best_image.category and best_image.category.description:
                return f"**Categoria:** {best_image.category.name}\n**Critério:** {best_image.category.description}\n**Similaridade visual:** {int(image_similarity * 100)}%\n**Correspondência com categoria:** {int(category_similarity * 100)}%"
            return f"Imagem classificada com base na similaridade visual ({int(image_similarity * 100)}%)."
    
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
