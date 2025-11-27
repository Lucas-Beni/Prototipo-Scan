# Prototipo-Scan

## Project Overview
API de busca de imagens por similaridade usando CLIP (Contrastive Language-Image Pre-Training) e FAISS (Facebook AI Similarity Search), com interface visual completa.

## Funcionalidades
- Interface visual para upload e busca de imagens
- Carrega modelo CLIP ViT-B/32 para geração de embeddings de imagens
- Indexa automaticamente todas as imagens na pasta `images/`
- Busca por similaridade de cosseno usando FAISS
- Exibe a imagem encontrada com percentual de similaridade
- 10 imagens de smartphones indexadas para testes

## Estrutura do Projeto
```
├── main.py               # Rotas da API Flask + servidor de arquivos
├── clip_engine.py        # Carregamento do modelo CLIP e geração de embeddings
├── indexer.py            # Indexação FAISS e mapeamento de imagens
├── utils.py              # Funções auxiliares
├── images/               # Pasta com imagens para indexar (smartphones)
├── templates/
│   └── index.html        # Interface visual principal
├── static/
│   ├── style.css         # Estilos da interface
│   └── script.js         # Lógica do frontend
├── requirements.txt      # Dependências Python
└── replit.md             # Este arquivo
```

## Interface Visual
A interface permite:
1. Upload de imagem por clique ou arrastar e soltar
2. Preview da imagem selecionada
3. Busca com um clique no botão
4. Exibição lado a lado: imagem enviada x imagem encontrada
5. Barra de similaridade com percentual
6. Grid de todas as imagens indexadas

## Endpoints da API

### GET /
Retorna a interface visual HTML.

### GET /api
Retorna informações sobre a API em JSON e lista de imagens indexadas.

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

### GET /images/<filename>
Serve as imagens indexadas.

### GET /health
Health check da API.

### POST /reindex
Força reindexação das imagens.

## Como Usar

### Via Interface Visual
1. Acesse a URL do projeto no navegador
2. Clique ou arraste uma imagem para o campo de upload
3. Clique em "Buscar Imagem Similar"
4. Veja o resultado com a imagem mais parecida e o percentual

### Via API (curl)
```bash
curl -X POST -F "image=@sua_imagem.jpg" http://localhost:5000/search
```

## Tecnologias
- Python 3.11
- Flask (API REST + Templates)
- CLIP (OpenAI) - Embeddings de imagens
- FAISS (Facebook) - Busca por similaridade vetorial
- PyTorch (CPU) - Backend para CLIP
- Pillow - Processamento de imagens
- HTML5/CSS3/JavaScript - Interface visual

## Notas Técnicas
- O modelo CLIP é carregado na inicialização do servidor
- Os embeddings são normalizados para uso com similaridade de cosseno
- FAISS usa IndexFlatIP (Inner Product) com vetores normalizados
- Otimizado para CPU (não requer GPU)
- Cache desabilitado para desenvolvimento
