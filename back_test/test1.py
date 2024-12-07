from transformers import pipeline
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
import os
import tensorflow as tf

# Шаг 1: Индексация документов

# Загрузка модели эмбеддингов
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Представление документов в виде векторов
documents = os.listdir("./documents")
print(documents)
document_embeddings = embedding_model.encode(documents)

# Создание индекса FAISS
index = IndexFlatIP(document_embeddings.shape[1])
index.add(document_embeddings)

# Шаг 2: Поиск релевантных документов

# Представление запроса в виде вектора
query = "first"
query_embedding = embedding_model.encode([query])

# Поиск ближайших соседей
distances, indices = index.search(query_embedding, k=2)
relevant_documents = [documents[i] for i in indices[0]]

# Шаг 3: Генерация ответа

# Подготовка контекста
context = "\n".join(relevant_documents)

# Загрузка модели генерации
generator = pipeline('text-generation', model='gpt-3')

# Генерация ответа
response = generator(f"Запрос: {query}\nКонтекст: {context}\nОтвет:")

print(response[0]['generated_text'])