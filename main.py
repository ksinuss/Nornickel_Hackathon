from __init__ import os, extract_text, cv2, pytesseract, Flask, render_template, request, jsonify
from algorithm import index_documents, search_documents

### format processing
def process_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    elif file_extension == '.pdf':
        content = extract_text(file_path)
    elif file_extension in ['.jpg', '.png', '.jpeg']:
        image = cv2.imread(file_path)
        content = pytesseract.image_to_string(image)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return content

### uploading documents
def load_documents(directory):
    documents = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        content = process_document(file_path)
        documents.append((file_name, content))
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
    # app.run(debug=True)
    app.run()
