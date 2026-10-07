import os
import time
import uuid

import streamlit as st
import chromadb

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Semantic Cache for LLM",
    page_icon="🧠",
    layout="wide"
)


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()


# =====================================================
# GEMINI CLIENT
# =====================================================

client = genai.Client(api_key=api_key)


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =====================================================
# CONNECT TO CHROMADB
# =====================================================

@st.cache_resource
def load_cache():

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    cache = chroma_client.get_or_create_collection(
        name="llm_cache",
        metadata={"hnsw:space": "cosine"}
    )

    return cache


cache = load_cache()


# =====================================================
# SESSION STATISTICS
# =====================================================

if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0

if "cache_misses" not in st.session_state:
    st.session_state.cache_misses = 0


# =====================================================
# TITLE
# =====================================================

st.title("🧠 Semantic Cache for LLM")

st.markdown(
    """
    Ask a question and the system will first search the
    semantic cache. If a sufficiently similar question
    already exists, the cached response is returned instead
    of calling the LLM.
    """
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Cache Configuration")

THRESHOLD = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Statistics")

total_requests = (
    st.session_state.cache_hits
    + st.session_state.cache_misses
)

st.sidebar.metric(
    "Total Requests",
    total_requests
)

st.sidebar.metric(
    "Cache Hits",
    st.session_state.cache_hits
)

st.sidebar.metric(
    "Cache Misses",
    st.session_state.cache_misses
)

if total_requests > 0:

    hit_rate = (
        st.session_state.cache_hits
        / total_requests
    ) * 100

else:

    hit_rate = 0


st.sidebar.metric(
    "Cache Hit Rate",
    f"{hit_rate:.2f}%"
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Cached Queries: **{cache.count()}**"
)


# =====================================================
# USER QUERY
# =====================================================

query = st.text_area(
    "Enter your query",
    placeholder="Example: Which city is the capital of Bihar?",
    height=120
)


# =====================================================
# ASK BUTTON
# =====================================================

if st.button(
    "🚀 Ask Question",
    use_container_width=True
):

    if not query.strip():

        st.warning("Please enter a query.")

    else:

        query = query.strip()

        # ---------------------------------------------
        # START TIMER
        # ---------------------------------------------

        start_time = time.perf_counter()


        # ---------------------------------------------
        # GENERATE EMBEDDING
        # ---------------------------------------------

        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()


        # ---------------------------------------------
        # CACHE SEARCH
        # ---------------------------------------------

        if cache.count() == 0:

            similarity = 0
            cache_hit = False

        else:

            results = cache.query(
                query_embeddings=[query_embedding],
                n_results=1
            )

            distance = results["distances"][0][0]

            similarity = 1 - distance

            cache_hit = similarity >= THRESHOLD


        # =================================================
        # CACHE HIT
        # =================================================

        if cache_hit:

            st.session_state.cache_hits += 1

            cached_query = results["documents"][0][0]

            cached_response = results["metadatas"][0][0][
                "response"
            ]


            # ---------------------------------------------
            # STOP TIMER
            # ---------------------------------------------

            end_time = time.perf_counter()

            latency = end_time - start_time


            # ---------------------------------------------
            # RESULT HEADER
            # ---------------------------------------------

            st.success("⚡ CACHE HIT")


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Similarity",
                    f"{similarity:.4f}"
                )

            with col2:

                st.metric(
                    "Response Time",
                    f"{latency:.4f} sec"
                )


            st.info(
                f"Matched cached query: **{cached_query}**"
            )


            st.subheader("💬 Answer")

            st.write(cached_response)


            st.caption(
                "Gemini API was not called. "
                "The response was retrieved from the semantic cache."
            )


        # =================================================
        # CACHE MISS
        # =================================================

        else:

            st.session_state.cache_misses += 1


            # ---------------------------------------------
            # CALL GEMINI
            # ---------------------------------------------

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=query
            )

            answer = response.text


            # ---------------------------------------------
            # STORE IN CACHE
            # ---------------------------------------------

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


            # ---------------------------------------------
            # STOP TIMER
            # ---------------------------------------------

            end_time = time.perf_counter()

            latency = end_time - start_time


            # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            st.warning("🔄 CACHE MISS")


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Similarity",
                    f"{similarity:.4f}"
                )

            with col2:

                st.metric(
                    "Response Time",
                    f"{latency:.4f} sec"
                )


            st.subheader("💬 Answer")

            st.write(answer)


            st.caption(
                "Gemini API was called and the new response "
                "was stored in the semantic cache."
            )


# =====================================================
# FOOTER / EXPLANATION
# =====================================================

st.markdown("---")

st.markdown(
    """
    ### How it works

    **User Query → Embedding → ChromaDB → Similarity Check**

    - 🟢 **Cache Hit:** Similar query found → return cached response
    - 🔴 **Cache Miss:** No sufficiently similar query → call Gemini → store response
    """
)