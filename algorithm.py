from . import RagTokenizer, RagRetriever, RagSequenceForGeneration, torch, cosine_similarity

### indexing documents using the RAG model
def index_documents(documents):
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    
    indexed_documents = []
    for file_name, content in documents:
        inputs = tokenizer(content, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs)
        vector = outputs.squeeze().numpy()
        indexed_documents.append((file_name, vector))
    
    return indexed_documents

### search using cosine similarity
def search_documents(indexed_documents, query):
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    
    query_inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_outputs = model.generate(**query_inputs)
    query_vector = query_outputs.squeeze().numpy()
    
    similarities = []
    for file_name, document_vector in indexed_documents:
        similarity = cosine_similarity([query_vector], [document_vector])[0][0]
        similarities.append((file_name, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities