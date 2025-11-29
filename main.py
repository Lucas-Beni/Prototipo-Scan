from flask import Flask, request, jsonify, render_template, send_from_directory
from models import db, Category, Image
from indexer import get_indexer, reset_indexer
from utils import load_image_from_bytes, validate_image_file
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prototipo_scan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

IMAGES_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

indexer = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_image_by_id(image_id):
    return Image.query.get(image_id)


def initialize_indexer():
    global indexer
    indexer = get_indexer()
    with app.app_context():
        images = Image.query.all()
        indexer.initialize_from_db(images)


@app.before_request
def ensure_initialized():
    global indexer
    if indexer is None:
        initialize_indexer()


@app.after_request
def add_cache_control(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('search.html')


@app.route('/categories')
def categories_page():
    return render_template('categories.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)


@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = Category.query.all()
    return jsonify([c.to_dict() for c in categories])


@app.route('/api/categories', methods=['POST'])
def create_category():
    data = request.get_json()
    
    if not data or not data.get('name'):
        return jsonify({'error': 'Nome da categoria é obrigatório'}), 400
    
    name = data['name'].strip()
    description = data.get('description', '').strip()
    
    existing = Category.query.filter_by(name=name).first()
    if existing:
        return jsonify({'error': 'Categoria com este nome já existe'}), 400
    
    category = Category(name=name, description=description)
    db.session.add(category)
    db.session.commit()
    
    return jsonify(category.to_dict()), 201


@app.route('/api/categories/<int:category_id>', methods=['GET'])
def get_category(category_id):
    category = Category.query.get_or_404(category_id)
    return jsonify(category.to_dict())


@app.route('/api/categories/<int:category_id>', methods=['PUT'])
def update_category(category_id):
    category = Category.query.get_or_404(category_id)
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Dados inválidos'}), 400
    
    if 'name' in data:
        name = data['name'].strip()
        existing = Category.query.filter(Category.name == name, Category.id != category_id).first()
        if existing:
            return jsonify({'error': 'Categoria com este nome já existe'}), 400
        category.name = name
    
    if 'description' in data:
        category.description = data['description'].strip()
    
    db.session.commit()
    return jsonify(category.to_dict())


@app.route('/api/categories/<int:category_id>', methods=['DELETE'])
def delete_category(category_id):
    category = Category.query.get_or_404(category_id)
    
    for image in category.images:
        if os.path.exists(image.storage_path):
            os.remove(image.storage_path)
    
    db.session.delete(category)
    db.session.commit()
    
    reset_indexer()
    initialize_indexer()
    
    return jsonify({'message': 'Categoria excluída com sucesso'})


@app.route('/api/images', methods=['GET'])
def get_images():
    category_id = request.args.get('category_id', type=int)
    
    if category_id:
        images = Image.query.filter_by(category_id=category_id).all()
    else:
        images = Image.query.all()
    
    return jsonify([img.to_dict() for img in images])


@app.route('/api/images', methods=['POST'])
def upload_image():
    global indexer
    
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['image']
    category_id = request.form.get('category_id', type=int)
    
    if not category_id:
        return jsonify({'error': 'Categoria é obrigatória'}), 400
    
    category = Category.query.get(category_id)
    if not category:
        return jsonify({'error': 'Categoria não encontrada'}), 404
    
    if file.filename == '':
        return jsonify({'error': 'Arquivo vazio'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
    
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    
    if not os.path.exists(IMAGES_FOLDER):
        os.makedirs(IMAGES_FOLDER)
    
    storage_path = os.path.join(IMAGES_FOLDER, unique_filename)
    file.save(storage_path)
    
    image_record = Image(
        filename=unique_filename,
        original_filename=original_filename,
        category_id=category_id,
        storage_path=storage_path
    )
    db.session.add(image_record)
    db.session.commit()
    
    if indexer and indexer.is_initialized:
        indexer.add_single_image(image_record)
    
    return jsonify(image_record.to_dict()), 201


@app.route('/api/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    image = Image.query.get_or_404(image_id)
    
    if os.path.exists(image.storage_path):
        os.remove(image.storage_path)
    
    db.session.delete(image)
    db.session.commit()
    
    reset_indexer()
    initialize_indexer()
    
    return jsonify({'message': 'Imagem excluída com sucesso'})


@app.route('/search', methods=['POST'])
def search():
    global indexer
    
    if 'image' not in request.files:
        return jsonify({
            'error': 'Nenhuma imagem enviada',
            'message': 'Envie uma imagem no campo "image"'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'error': 'Arquivo vazio',
            'message': 'O arquivo enviado está vazio'
        }), 400
    
    try:
        image_bytes = file.read()
        
        if not validate_image_file(image_bytes):
            return jsonify({
                'error': 'Arquivo inválido',
                'message': 'O arquivo enviado não é uma imagem válida'
            }), 400
        
        query_image = load_image_from_bytes(image_bytes)
        
        if indexer.get_indexed_count() == 0:
            return jsonify({
                'error': 'Índice vazio',
                'message': 'Não há imagens indexadas. Adicione imagens primeiro.'
            }), 404
        
        image_record, similarity, percentage = indexer.search(query_image, get_image_by_id)
        
        if image_record is None:
            return jsonify({
                'error': 'Nenhuma correspondência encontrada',
                'message': 'Não foi possível encontrar uma imagem similar'
            }), 404
        
        return jsonify({
            'match_image': image_record.filename,
            'original_filename': image_record.original_filename,
            'similarity': round(similarity, 4),
            'percentage': percentage,
            'category': {
                'id': image_record.category.id,
                'name': image_record.category.name,
                'description': image_record.category.description
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500


@app.route('/api/stats')
def stats():
    return jsonify({
        'indexed_images': indexer.get_indexed_count() if indexer and indexer.is_initialized else 0,
        'total_categories': Category.query.count(),
        'total_images': Image.query.count()
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'indexed_images': indexer.get_indexed_count() if indexer and indexer.is_initialized else 0
    })


@app.route('/reindex', methods=['POST'])
def reindex():
    global indexer
    try:
        reset_indexer()
        initialize_indexer()
        return jsonify({
            'status': 'success',
            'message': 'Reindexação concluída',
            'indexed_images': indexer.get_indexed_count()
        })
    except Exception as e:
        return jsonify({
            'error': 'Erro na reindexação',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("=" * 60)
        print("PROTOTIPO-SCAN API")
        print("Sistema de busca de imagens com categorias")
        print("=" * 60)
        
        initialize_indexer()
        
        print("=" * 60)
        print(f"Imagens indexadas: {indexer.get_indexed_count()}")
        print(f"Categorias: {Category.query.count()}")
        print("Servidor iniciando em http://0.0.0.0:5000")
        print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
