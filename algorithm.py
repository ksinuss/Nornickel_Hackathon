from __init__ import os, index, Schema, TEXT, ID, QueryParser, extract_text, Image, pytesseract, torch, AutoModel, AutoTokenizer, pickle

# Определение схемы индекса
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True))

# Создание индекса
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
ix = index.create_in("indexdir", schema)

### indexing documents using the RAG model
def index_documents(documents):
    writer = ix.writer()
    for title, content in documents:
        writer.add_document(title=title, path=title, content=content)
    writer.commit()

### search using cosine similarity
def search_documents(documents, query):
    ix = index.open_dir("indexdir")
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query)
        results = searcher.search(query)
        return [result['title'] for result in results]
    
def create_local_dataset(documents):
    local_dataset = []
    for title, content in documents:
        local_dataset.append((title, content))
    
    with open('local_dataset.pkl', 'wb') as f:
        pickle.dump(local_dataset, f)

# Загрузка модели ColPali
tokenizer = AutoTokenizer.from_pretrained("path_to_colpali_tokenizer")
model = AutoModel.from_pretrained("path_to_colpali_model")

def enhance_search_with_colpali(query):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    enhanced_query = tokenizer.decode(outputs.last_hidden_state[0].argmax(dim=-1))
    return enhanced_query