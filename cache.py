import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------
# 1. Load embedding model
# --------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 2. Connect to ChromaDB
# --------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# --------------------------------
# 3. Create cosine-distance collection
# --------------------------------

cache = chroma_client.get_or_create_collection(
    name="semantic_cache_cosine",
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------
# 4. Store original query
# --------------------------------

original_query = "What is machine learning?"

original_embedding = embedding_model.encode(
    original_query,
    normalize_embeddings=True
).tolist()


cache.add(
    ids=["query_1"],
    embeddings=[original_embedding],
    documents=[original_query],
    metadatas=[
        {
            "response": "Machine learning is a branch of artificial intelligence that allows computers to learn from data."
        }
    ]
)


# --------------------------------
# 5. New query
# --------------------------------

new_query = "Explain machine learning"

new_embedding = embedding_model.encode(
    new_query,
    normalize_embeddings=True
).tolist()


# --------------------------------
# 6. Search cache
# --------------------------------

results = cache.query(
    query_embeddings=[new_embedding],
    n_results=1
)


# --------------------------------
# 7. Get result
# --------------------------------

matched_query = results["documents"][0][0]
distance = results["distances"][0][0]

# Cosine distance = 1 - cosine similarity
similarity = 1 - distance


print("New Query:")
print(new_query)

print("\nMatched Cached Query:")
print(matched_query)

print("\nCosine Similarity:")
print(similarity)

THRESHOLD = 0.80

if similarity >= THRESHOLD:
    print("\nCACHE HIT")
    print("Returning cached response...")

    cached_response = results["metadatas"][0][0]["response"]

    print("\nCached Response:")
    print(cached_response)

else:
    print("\nCACHE MISS")
    print("Need to call Gemini.")