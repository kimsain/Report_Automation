# rag_system_v2.py (수정본)
# 결과: embedding_store DB가 존재하고, 매번 생성 필요 없음

import os
import torch
import faiss
import psycopg2

import numpy as np

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# 환경 변수 로드
load_dotenv()

# DB 연결 정보
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# DB 연결
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

query_text = "0지역의 축제 기간 동안 가장 많이 방문한 연령층과 가장 적게 방문한 연령층을 알려줘"
top_k = 10
# If question not provided as an argument, prompt the user for input
if query_text is None:
    query_text = input("Enter your question: ").strip()
# If top_k was not provided (argparse will have default=5 if not given), ensure it's an integer
try:
    top_k = int(top_k)
except Exception:
    top_k = int(input("Enter the number of top results to retrieve: ").strip())

# Load the KoSimCSE model and tokenizer from Hugging Face
# (This will download the model on first run if not cached)
tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
model.eval()  # set model to evaluation mode (no dropout)

# (Optional: you can move the model to GPU for faster inference if available)
# if torch.cuda.is_available():
#     model.to("cuda")

# Tokenize the input query and get the embedding from the model
inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():  # no gradient needed for inference
    outputs = model(**inputs, return_dict=False)
# The model returns a tuple: (last_hidden_state, pooler_output). We take the [CLS] token embedding from last_hidden_state.
last_hidden_state = outputs[0]            # shape: (batch_size=1, seq_len, 768)
cls_embedding = last_hidden_state[0, 0]   # take the first token ([CLS]) of the first (and only) sequence
# Convert embedding to a Python list (for SQL query). Ensure it’s a list of floats.
embedding_vector = cls_embedding.cpu().numpy().tolist()  # 768-dim vector

# Formulate the SQL query to perform vector similarity search using the pgvector <=> operator (cosine distance).
# We cast the embedding list to the vector type in SQL by using an array literal and ::vector.
# 1 - (embedding <=> query_vector) gives the cosine similarity score.
query_placeholder = "%s"  # we'll pass our vector and top_k as parameters
sql = (
    "SELECT original_text, 1 - (embedding <=> " + query_placeholder + "::vector) AS similarity "
    "FROM embedding_store "
    "ORDER BY embedding <=> " + query_placeholder + "::vector "
    "LIMIT " + query_placeholder
)

try:
    # Execute the similarity search query, passing the embedding vector twice (for SELECT and ORDER BY) and top_k.
    cursor.execute(sql, (embedding_vector, embedding_vector, top_k))
    results = cursor.fetchall()
except Exception as e:
    print("Error querying the database:", e)
    results = []
finally:
    # Close the database cursor and connection
    cursor.close()
    conn.close()

# Output the results
if results:
    print(f"\nTop {top_k} results for query: \"{query_text}\":")
    for i, (text, similarity) in enumerate(results, start=1):
        # similarity is cosine similarity (range -1 to 1). We format it as percentage or float.
        print(f"{i}. {text}  (Similarity: {similarity:.4f})")
else:
    print("No results found. Make sure the database is populated and query is valid.")