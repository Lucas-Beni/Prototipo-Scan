# Prototipo-Scan

## Project Overview
API de busca de imagens por similaridade usando CLIP (Contrastive Language-Image Pre-Training) e FAISS (Facebook AI Similarity Search).

## Funcionalidades
- Carrega modelo CLIP ViT-B/32 para geração de embeddings de imagens
- Indexa automaticamente todas as imagens na pasta `images/`
- Busca por similaridade de cosseno usando FAISS
- Retorna a imagem mais similar com score de similaridade

## Estrutura do Projeto
```
├── main.py           # Rotas da API Flask
├── clip_engine.py    # Carregamento do modelo CLIP e geração de embeddings
├── indexer.py        # Indexação FAISS e mapeamento de imagens
├── utils.py          # Funções auxiliares
├── images/           # Pasta com imagens para indexar
├── requirements.txt  # Dependências Python
└── replit.md         # Este arquivo
```

## Endpoints da API

### GET /
Retorna informações sobre a API e lista de imagens indexadas.

### POST /search
Busca a imagem mais similar. Envie uma imagem no campo `image` (form-data).

**Resposta:**
```json
{
  "match_image": "nome_do_arquivo.jpg",
  "similarity": 0.85,
  "percentage": 85
}
```

### GET /health
Health check da API.

### POST /reindex
Força reindexação das imagens.

## Como Usar

1. Adicione imagens à pasta `images/`
2. Inicie o servidor (workflow automático)
3. Faça uma requisição POST para `/search` com uma imagem
4. Receba o resultado com a imagem mais similar

### Exemplo com curl:
```bash
curl -X POST -F "image=@sua_imagem.jpg" http://localhost:5000/search
```

## Tecnologias
- Python 3.11
- Flask (API REST)
- CLIP (OpenAI) - Embeddings de imagens
- FAISS (Facebook) - Busca por similaridade vetorial
- PyTorch (CPU) - Backend para CLIP
- Pillow - Processamento de imagens

## Notas Técnicas
- O modelo CLIP é carregado na inicialização do servidor
- Os embeddings são normalizados para uso com similaridade de cosseno
- FAISS usa IndexFlatIP (Inner Product) com vetores normalizados
- Otimizado para CPU (não requer GPU)
