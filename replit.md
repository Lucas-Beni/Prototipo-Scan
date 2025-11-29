# Prototipo-Scan

## Project Overview
Sistema de busca de imagens por similaridade usando CLIP (Contrastive Language-Image Pre-Training) e FAISS (Facebook AI Similarity Search), com suporte a categorização de imagens e interface visual completa.

## Funcionalidades
- Interface visual para upload e busca de imagens
- Sistema de categorias para classificação de imagens (ex: Leve, Médio, Pesado)
- Cadastro de imagens associadas a categorias
- Busca por similaridade retorna imagem + categoria
- Carrega modelo CLIP ViT via HuggingFace API para geração de embeddings
- Indexa automaticamente todas as imagens do banco de dados
- Busca por similaridade de cosseno usando FAISS
- Exibe a imagem encontrada com percentual de similaridade e categoria

## Estrutura do Projeto
```
├── main.py               # Rotas da API Flask + servidor de arquivos
├── models.py             # Modelos SQLAlchemy (Category, Image)
├── clip_engine.py        # Geração de embeddings via HuggingFace API
├── indexer.py            # Indexação FAISS e mapeamento de imagens
├── utils.py              # Funções auxiliares
├── images/               # Pasta com imagens cadastradas
├── templates/
│   ├── search.html       # Tela de busca de imagens
│   ├── categories.html   # Tela de gerenciamento de categorias
│   └── upload.html       # Tela de cadastro de imagens
├── static/
│   └── style.css         # Estilos da interface
├── instance/
│   └── prototipo_scan.db # Banco de dados SQLite
├── requirements.txt      # Dependências Python
└── replit.md             # Este arquivo
```

## Telas do Sistema

### 1. Busca (/)
- Upload de imagem por clique ou arrastar e soltar
- Preview da imagem selecionada
- Busca com um clique no botão
- Exibição lado a lado: imagem enviada x imagem encontrada
- Exibe categoria da imagem encontrada
- Barra de similaridade com percentual
- Estatísticas do sistema

### 2. Categorias (/categories)
- Formulário para criar novas categorias
- Lista de categorias cadastradas com contagem de imagens
- Editar e excluir categorias
- Exclusão em cascata (remove imagens associadas)

### 3. Cadastrar Imagens (/upload)
- Seleção de categoria obrigatória
- Upload de imagem com preview
- Filtro de imagens por categoria
- Grid de imagens cadastradas
- Exclusão individual de imagens

## Endpoints da API

### Páginas
- `GET /` - Tela de busca de imagens
- `GET /categories` - Tela de gerenciamento de categorias
- `GET /upload` - Tela de cadastro de imagens

### API de Categorias
- `GET /api/categories` - Lista todas as categorias
- `POST /api/categories` - Cria nova categoria
- `GET /api/categories/<id>` - Retorna categoria específica
- `PUT /api/categories/<id>` - Atualiza categoria
- `DELETE /api/categories/<id>` - Exclui categoria

### API de Imagens
- `GET /api/images` - Lista imagens (opcional: ?category_id=X)
- `POST /api/images` - Upload de imagem com categoria
- `DELETE /api/images/<id>` - Exclui imagem

### Busca
- `POST /search` - Busca imagem similar

**Resposta:**
```json
{
  "match_image": "uuid_nome_do_arquivo.jpg",
  "original_filename": "nome_original.jpg",
  "similarity": 0.85,
  "percentage": 85,
  "category": {
    "id": 1,
    "name": "Pesado",
    "description": "Categoria para itens pesados"
  }
}
```

### Utilitários
- `GET /api/stats` - Estatísticas do sistema
- `GET /health` - Health check
- `POST /reindex` - Força reindexação

## Como Usar

### Fluxo Recomendado
1. Acesse `/categories` e crie as categorias desejadas
2. Acesse `/upload` e cadastre imagens associando-as às categorias
3. Acesse `/` para buscar imagens por similaridade

### Via API (curl)
```bash
# Criar categoria
curl -X POST -H "Content-Type: application/json" \
  -d '{"name": "Pesado", "description": "Itens pesados"}' \
  http://localhost:5000/api/categories

# Upload de imagem
curl -X POST -F "image=@sua_imagem.jpg" -F "category_id=1" \
  http://localhost:5000/api/images

# Buscar imagem similar
curl -X POST -F "image=@imagem_busca.jpg" \
  http://localhost:5000/search
```

## Tecnologias
- Python 3.11
- Flask + Flask-SQLAlchemy
- SQLite (banco de dados)
- HuggingFace Inference API (embeddings de imagens)
- FAISS (busca por similaridade vetorial)
- Pillow (processamento de imagens)
- HTML5/CSS3/JavaScript (interface visual)

## Variáveis de Ambiente
- `HF_API_KEY` - Chave de API do HuggingFace (obrigatória)

## Notas Técnicas
- Os embeddings são gerados via HuggingFace Inference API (modelo google/vit-base-patch16-224)
- Os embeddings são normalizados para uso com similaridade de cosseno
- FAISS usa IndexFlatIP (Inner Product) com vetores normalizados
- Banco de dados SQLite para persistência de categorias e metadados de imagens
- Cache desabilitado para desenvolvimento
- Imagens são armazenadas com UUID para evitar conflitos de nomes

## Recent Changes
- **29/Nov/2025**: Adicionado sistema de categorias e cadastro de imagens
  - Nova tela de gerenciamento de categorias
  - Nova tela de cadastro de imagens com associação a categorias
  - Busca agora retorna a categoria da imagem encontrada
  - Banco de dados SQLite para persistência
  - Navegação entre as três telas
