import time
import os
from dotenv import load_dotenv
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import voyageai

load_dotenv()

# --- Initialisation des clients ---

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="fabric_docs",
    embedding_function=embedding_fn
)

claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# --- Construction de l'index BM25 au démarrage ---
# BM25 a besoin de TOUS les chunks dès le début (contrairement à ChromaDB qui cherche à la demande)
# On charge donc tout en mémoire une seule fois ici

print("📦 Chargement de tous les chunks pour BM25...")
all_data = collection.get()
all_chunks = all_data["documents"]
all_ids = all_data["ids"]
all_metadatas = all_data["metadatas"]

# BM25 tokenise chaque chunk en liste de mots
tokenized_corpus = [chunk.split() for chunk in all_chunks]
bm25_index = BM25Okapi(tokenized_corpus)
print(f"✅ Index BM25 prêt ({len(all_chunks)} chunks)")


def retrieve_hybrid(question, n_candidates=10):
    """
    Recherche hybride : sémantique (ChromaDB) + keyword (BM25).
    Retourne jusqu'à 2×n_candidates candidats uniques avant reranking.

    Pourquoi les deux ?
    - Sémantique : trouve les chunks conceptuellement proches, même sans mot exact
    - BM25 : indispensable quand la question contient un terme technique précis
      (ex : "OneLake" → BM25 le trouve immédiatement, le sémantique peut rater)
    """

    # --- Recherche sémantique via ChromaDB ---
    semantic_results = collection.query(
        query_texts=[question],
        n_results=n_candidates
    )
    semantic_chunks = semantic_results["documents"][0]
    semantic_ids = semantic_results["ids"][0]
    semantic_metas = semantic_results["metadatas"][0]

    # --- Recherche BM25 (keyword) ---
    tokenized_query = question.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # On récupère les indices des n_candidates meilleurs scores BM25
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:n_candidates]

    bm25_chunks = [all_chunks[i] for i in top_bm25_indices]
    bm25_ids = [all_ids[i] for i in top_bm25_indices]
    bm25_metas = [all_metadatas[i] for i in top_bm25_indices]

    # --- Fusion RRF (Reciprocal Rank Fusion) ---
    # Chaque chunk reçoit un score = 1/(k + rang) dans chaque liste
    # Un chunk présent dans les deux listes cumule les deux scores → il remonte
    # k=60 est la valeur standard de la littérature (Cormack et al. 2009)

    k = 60
    rrf_scores = {}   # chunk_id → score RRF cumulé
    chunk_store = {}  # chunk_id → (texte, metadata) pour retrouver le contenu

    for rank, (chunk_id, chunk, meta) in enumerate(zip(semantic_ids, semantic_chunks, semantic_metas)):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_store[chunk_id] = (chunk, meta)

    for rank, (chunk_id, chunk, meta) in enumerate(zip(bm25_ids, bm25_chunks, bm25_metas)):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_store[chunk_id] = (chunk, meta)

    # On trie par score RRF décroissant et on reconstruit les listes
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    merged_chunks = [chunk_store[i][0] for i in sorted_ids]
    merged_metas = [chunk_store[i][1] for i in sorted_ids]

    return merged_chunks, merged_metas


def rerank(question, chunks, metadatas, top_k=5):
    time.sleep(20)
    """
    Reranking avec Voyage AI.

    Pourquoi le reranking change tout :
    - ChromaDB + BM25 utilisent des scores indépendants (vecteur vs fréquence de mots)
    - Le reranker est un modèle cross-encoder : il lit la question ET le chunk ENSEMBLE
    - C'est beaucoup plus précis mais trop lent pour chercher sur 1000 chunks
    - D'où le pattern candidats larges (20) → reranker précis → top 5
    """
    result = voyage_client.rerank(
        query=question,
        documents=chunks,
        model="rerank-2",
        top_k=top_k
    )

    # result.results est une liste d'objets avec .index et .relevance_score
    reranked_chunks = [chunks[r.index] for r in result.results]
    reranked_metas = [metadatas[r.index] for r in result.results]

    return reranked_chunks, reranked_metas


def generate(question, chunks, metadatas):
    """
    Identique à query.py v1 — on passe les chunks retenus à Claude.
    """
    context = ""
    for chunk, meta in zip(chunks, metadatas):
        context += f"[Source : {meta['source']}]\n{chunk}\n\n"

    prompt = f"""Tu es un assistant expert Microsoft Fabric.
Réponds à la question en te basant uniquement sur le contexte ci-dessous.
Si la réponse n'est pas dans le contexte, dis-le clairement.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def ask(question):
    """
    Pipeline RAG v2 complet :
    1. retrieve_hybrid() → sémantique + BM25 + fusion RRF → ~20 candidats
    2. rerank()          → Voyage AI réduit à top 5
    3. generate()        → Claude génère la réponse
    """
    print(f"\n❓ Question : {question}")
    print("🔍 Recherche hybride (sémantique + BM25)...")

    chunks, metadatas = retrieve_hybrid(question)
    print(f"   → {len(chunks)} candidats après fusion RRF")

    print("🎯 Reranking avec Voyage AI...")
    chunks, metadatas = rerank(question, chunks, metadatas, top_k=5)

    sources = list(set(m["source"] for m in metadatas))
    print(f"📄 Sources retenues : {sources}")

    print("🤖 Génération de la réponse avec Claude...")
    answer = generate(question, chunks, metadatas)

    print(f"\n💬 Réponse :\n{answer}")
    print("\n" + "─"*60)
    return answer


if __name__ == "__main__":
    questions = [
        "Qu'est-ce que Microsoft Fabric ?",
        "Quelle est la différence entre Direct Lake et Import mode ?",
        "Comment fonctionne OneLake ?",
        "Quels sont les avantages de Direct Lake pour les grands volumes de données ?",
        "Qu'est-ce que Data Factory dans Microsoft Fabric ?"
    ]

    for question in questions:
        ask(question)