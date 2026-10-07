import os
import time
import uuid
import chromadb

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# ==========================================
# 2. Create Gemini client
# ==========================================

client = genai.Client(api_key=api_key)


# ==========================================
# 3. Load embedding model
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 4. Connect to ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ==========================================
# 5. Create / load semantic cache
# ==========================================

cache = chroma_client.get_or_create_collection(
    name="llm_cache",
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 6. Configuration
# ==========================================

THRESHOLD = 0.80

cache_hits = 0
cache_misses = 0


# ==========================================
# 7. Application
# ==========================================

print("\n" + "=" * 50)
print("        SEMANTIC CACHE FOR LLM")
print("=" * 50)

print("\nType 'exit' to stop the application.")


while True:

    # --------------------------------------
    # Get query from user
    # --------------------------------------

    query = input("\nEnter your query: ").strip()


    # --------------------------------------
    # Exit condition
    # --------------------------------------

    if query.lower() == "exit":
        break


    # Ignore empty queries
    if not query:
        print("Please enter a valid query.")
        continue


    # Start timer
    start_time = time.perf_counter()


    # --------------------------------------
    # Generate embedding
    # --------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    # --------------------------------------
    # Check if cache contains anything
    # --------------------------------------

    if cache.count() == 0:

        cache_misses += 1

        print("\nCACHE MISS")


        # Call Gemini
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=query
        )

        answer = response.text


        print("\nGemini Response:")
        print(answer)


        # Store response in cache
        cache.add(
            ids=[str(uuid.uuid4())],
            embeddings=[query_embedding],
            documents=[query],
            metadatas=[
                {
                    "response": answer
                }
            ]
        )


        print("\nResponse stored in cache.")


    else:

        # --------------------------------------
        # Search semantic cache
        # --------------------------------------

        results = cache.query(
            query_embeddings=[query_embedding],
            n_results=1
        )


        distance = results["distances"][0][0]

        similarity = 1 - distance


        print("\nSimilarity:", round(similarity, 4))


        # --------------------------------------
        # CACHE HIT
        # --------------------------------------

        if similarity >= THRESHOLD:

            cache_hits += 1

            print("CACHE HIT")


            cached_query = results["documents"][0][0]

            cached_response = results["metadatas"][0][0]["response"]


            print("\nMatched Cached Query:")
            print(cached_query)


            print("\nCached Response:")
            print(cached_response)


        # --------------------------------------
        # CACHE MISS
        # --------------------------------------

        else:

            cache_misses += 1

            print("CACHE MISS")


            # Call Gemini
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=query
            )

            answer = response.text


            print("\nGemini Response:")
            print(answer)


            # Store new response
            cache.add(
                ids=[str(uuid.uuid4())],
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[
                    {
                        "response": answer
                    }
                ]
            )


            print("\nResponse stored in cache.")


    # --------------------------------------
    # Calculate response time
    # --------------------------------------

    end_time = time.perf_counter()

    latency = end_time - start_time


    print(
        "\nResponse Time:",
        round(latency, 4),
        "seconds"
    )


# ==========================================
# 8. Final statistics
# ==========================================

total_requests = cache_hits + cache_misses


print("\n" + "=" * 50)
print("             STATISTICS")
print("=" * 50)

print("Total Requests :", total_requests)
print("Cache Hits     :", cache_hits)
print("Cache Misses   :", cache_misses)


if total_requests > 0:

    hit_rate = (cache_hits / total_requests) * 100

    print(
        "Cache Hit Rate :",
        round(hit_rate, 2),
        "%"
    )

else:

    print("Cache Hit Rate : 0%")


print("=" * 50)
print("Application closed.")