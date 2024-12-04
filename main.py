from __init__ import os, extract_text, cv2, pytesseract, Flask, render_template, request, jsonify, Elasticsearch, Document
from algorithm import index_documents, search_documents

### format processing
def process_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    elif file_extension == '.pdf':
        try:
            content = extract_text(file_path)
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {file_path}") from e
    elif file_extension in ['.jpg', '.png', '.jpeg']:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Invalid image file: {file_path}")
        content = pytesseract.image_to_string(image)
    elif file_extension == '.docx':
        doc = Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return content

### uploading documents
def load_documents(directory):
    documents = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            content = process_document(file_path)
            documents.append((file_name, content))
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    return documents

documents_dir = 'documents'
app = Flask(__name__)

### main pipeline
def indexed(documents_dir):
    documents = load_documents(documents_dir)
    indexed_documents = index_documents(documents)
    return indexed_documents

### launching flask application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_files', methods=['GET'])
def get_files():
    files = os.listdir(documents_dir)
    return jsonify(files)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_documents(indexed(documents_dir), query)
    return jsonify(results)

if __name__ == '__main__':
    app.run()
