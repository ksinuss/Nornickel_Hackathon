from . import os, extract_text, cv2, pytesseract
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

### main pipeline
def main():
    documents_dir = 'path/to/documents'
    documents = load_documents(documents_dir)
    
    indexed_documents = index_documents(documents)
    
    query = input("Введите запрос для поиска: ")
    results = search_documents(indexed_documents, query)
    
    print("Результаты поиска:")
    for file_name, similarity in results:
        print(f"Файл: {file_name}, Сходство: {similarity:.4f}")

if __name__ == "__main__":
    main()
