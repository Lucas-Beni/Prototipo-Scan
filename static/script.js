document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const clearBtn = document.getElementById('clearBtn');
    const searchBtn = document.getElementById('searchBtn');
    const resultSection = document.getElementById('resultSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const indexedGrid = document.getElementById('indexedGrid');
    const indexedCount = document.getElementById('indexedCount');

    let selectedFile = null;

    loadIndexedImages();

    uploadArea.addEventListener('click', () => imageInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    clearBtn.addEventListener('click', clearSelection);
    
    searchBtn.addEventListener('click', performSearch);

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Por favor, selecione um arquivo de imagem.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadArea.style.display = 'none';
            searchBtn.disabled = false;
        };
        
        reader.readAsDataURL(file);
    }

    function clearSelection() {
        selectedFile = null;
        imageInput.value = '';
        previewContainer.style.display = 'none';
        uploadArea.style.display = 'block';
        searchBtn.disabled = true;
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';
    }

    async function performSearch() {
        if (!selectedFile) return;

        const btnText = searchBtn.querySelector('.btn-text');
        const btnLoading = searchBtn.querySelector('.btn-loading');
        
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        searchBtn.disabled = true;
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/search', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                showError(data.message || 'Erro ao buscar imagem similar.');
            }
        } catch (error) {
            showError('Erro de conexão. Tente novamente.');
        } finally {
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
            searchBtn.disabled = false;
        }
    }

    function showResult(data) {
        document.getElementById('sentImage').src = previewImage.src;
        document.getElementById('matchedImage').src = `/images/${data.match_image}`;
        document.getElementById('matchFilename').textContent = data.match_image;
        
        const percentage = data.percentage;
        document.getElementById('similarityValue').textContent = `${percentage}%`;
        document.getElementById('similarityBar').style.width = `${percentage}%`;
        
        resultSection.style.display = 'block';
        errorSection.style.display = 'none';
        
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        resultSection.style.display = 'none';
    }

    async function loadIndexedImages() {
        try {
            const response = await fetch('/api');
            const data = await response.json();
            
            indexedCount.textContent = data.indexed_images;
            indexedGrid.innerHTML = '';
            
            data.indexed_files.forEach(filename => {
                const item = document.createElement('div');
                item.className = 'indexed-item';
                item.innerHTML = `<img src="/images/${filename}" alt="${filename}" loading="lazy">`;
                indexedGrid.appendChild(item);
            });
        } catch (error) {
            console.error('Erro ao carregar imagens indexadas:', error);
        }
    }
});
