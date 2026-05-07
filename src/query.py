import os
from dotenv import load_dotenv
import anthropic
import chromadb
from chromadb.utils import embedding_functions

# Charge les variables d'environnement depuis .env

load_dotenv()

# --- Initialisation des clients ---
# IMPORTANT : on utilise exactement le même modèle d'embedding que dans ingest.py
# Si on changeait de modèle ici, les vecteurs seraient dans des espaces différents
# et la recherche retournerait des résultats complètement incohérents

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# On se connecte à la base existante — get_collection (pas get_or_create)
# car la base DOIT déjà exister (créée par ingest.py). Si elle n'existe pas, on veut une erreur claire.

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="fabric_docs",
    embedding_function=embedding_fn
)

# Claude joue le rôle du "G" dans RAG (Generate)
# Il ne cherche pas dans les docs — il reçoit les chunks déjà trouvés et rédige une réponse

claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def retrieve(question, n_results=5):
    """
    C'est le "R" du RAG (Retrieve).
    ChromaDB convertit la question en vecteur, puis cherche les 5 chunks
    dont le vecteur est le plus proche — c'est la similarité cosinus sous le capot.
    On récupère aussi les métadonnées pour savoir de quel PDF vient chaque chunk.
    """
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0], results["metadatas"][0]


def generate(question, chunks, metadatas):
    """
    C'est le "A+G" du RAG (Augment + Generate).
    On injecte les chunks dans le prompt de Claude — c'est l'augmentation.
    Claude n'invente pas : il est contraint de répondre depuis ce contexte uniquement.
    Cette contrainte ("uniquement sur le contexte") est ce qui évite les hallucinations.
    """

    # On formate les chunks en blocs lisibles avec leur source
    # Garder la source visible aide Claude à ne pas mélanger les docs entre eux

    context = ""
    for chunk, meta in zip(chunks, metadatas):
        context += f"[Source : {meta['source']}]\n{chunk}\n\n"

    # Le system prompt est minimaliste mais les deux instructions clés sont :
    # 1. "uniquement sur le contexte" → pas d'hallucination
    # 2. "si la réponse n'est pas dans le contexte, dis-le" → honnêteté sur les limites

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
    Orchestre le pipeline RAG complet en 2 étapes :
    1. retrieve() → trouve les chunks pertinents dans ChromaDB
    2. generate() → passe ces chunks à Claude pour construire la réponse

    C'est volontairement séparé en deux fonctions : ça permet plus tard
    d'améliorer chaque étape indépendamment (ex : meilleur reranking avant generate,
    ou meilleur prompt dans generate) sans tout réécrire.
    """

    print(f"\n❓ Question : {question}")
    print("🔍 Recherche dans la base vectorielle...")

    chunks, metadatas = retrieve(question)

    # On déduplique les sources pour l'affichage (un même PDF peut apparaître plusieurs fois)

    sources = list(set(m["source"] for m in metadatas))
    print(f"📄 Sources trouvées : {sources}")

    print("🤖 Génération de la réponse avec Claude...")
    answer = generate(question, chunks, metadatas)

    print(f"\n💬 Réponse :\n{answer}")
    print("\n" + "─"*60)
    return answer


if __name__ == "__main__":

    # 5 questions d'exemple couvrant les 3 PDFs indexés
    # Ces questions deviennent aussi les exemples du README

    questions = [
        "Qu'est-ce que Microsoft Fabric ?",
        "Quelle est la différence entre Direct Lake et Import mode ?",
        "Comment fonctionne OneLake ?",
        "Quels sont les avantages de Direct Lake pour les grands volumes de données ?",
        "Qu'est-ce que Data Factory dans Microsoft Fabric ?"
    ]

    for question in questions:
        ask(question)