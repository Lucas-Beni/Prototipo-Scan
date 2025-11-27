from flask import Flask, request, jsonify
from indexer import get_indexer
from utils import load_image_from_bytes, validate_image_file

app = Flask(__name__)

indexer = None


@app.before_request
def ensure_initialized():
    """
    Garante que o indexador está inicializado antes de processar requisições.
    """
    global indexer
    if indexer is None:
        indexer = get_indexer()
        indexer.initialize()


@app.route('/')
def index():
    """
    Rota principal - retorna informações sobre a API.
    """
    indexed_count = indexer.get_indexed_count() if indexer and indexer.is_initialized else 0
    indexed_files = indexer.get_all_indexed_files() if indexer and indexer.is_initialized else []
    
    return jsonify({
        "api": "Prototipo-Scan API",
        "version": "1.0.0",
        "description": "API de busca de imagens por similaridade usando CLIP e FAISS",
        "endpoints": {
            "POST /search": "Busca a imagem mais similar. Envie uma imagem no campo 'image'."
        },
        "indexed_images": indexed_count,
        "indexed_files": indexed_files
    })


@app.route('/search', methods=['POST'])
def search():
    """
    Rota de busca - recebe uma imagem e retorna a mais similar do índice.
    
    Espera um arquivo de imagem no campo 'image' do form-data.
    
    Returns:
        JSON com match_image, similarity e percentage
    """
    if 'image' not in request.files:
        return jsonify({
            "error": "Nenhuma imagem enviada",
            "message": "Envie uma imagem no campo 'image'"
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            "error": "Arquivo vazio",
            "message": "O arquivo enviado está vazio"
        }), 400
    
    try:
        image_bytes = file.read()
        
        if not validate_image_file(image_bytes):
            return jsonify({
                "error": "Arquivo inválido",
                "message": "O arquivo enviado não é uma imagem válida"
            }), 400
        
        query_image = load_image_from_bytes(image_bytes)
        
        if indexer.get_indexed_count() == 0:
            return jsonify({
                "error": "Índice vazio",
                "message": "Não há imagens indexadas. Adicione imagens à pasta 'images/' e reinicie o servidor."
            }), 404
        
        match_filename, similarity, percentage = indexer.search(query_image)
        
        if match_filename is None:
            return jsonify({
                "error": "Nenhuma correspondência encontrada",
                "message": "Não foi possível encontrar uma imagem similar"
            }), 404
        
        return jsonify({
            "match_image": match_filename,
            "similarity": round(similarity, 4),
            "percentage": percentage
        })
        
    except Exception as e:
        return jsonify({
            "error": "Erro interno",
            "message": str(e)
        }), 500


@app.route('/health')
def health():
    """
    Rota de health check.
    """
    return jsonify({
        "status": "healthy",
        "indexed_images": indexer.get_indexed_count() if indexer and indexer.is_initialized else 0
    })


@app.route('/reindex', methods=['POST'])
def reindex():
    """
    Força reindexação das imagens.
    """
    global indexer
    try:
        indexer = get_indexer()
        indexer.initialize()
        return jsonify({
            "status": "success",
            "message": "Reindexação concluída",
            "indexed_images": indexer.get_indexed_count()
        })
    except Exception as e:
        return jsonify({
            "error": "Erro na reindexação",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("PROTOTIPO-SCAN API")
    print("Busca de imagens por similaridade usando CLIP + FAISS")
    print("=" * 60)
    
    indexer = get_indexer()
    indexer.initialize()
    
    print("=" * 60)
    print(f"Imagens indexadas: {indexer.get_indexed_count()}")
    print("Servidor iniciando em http://0.0.0.0:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
