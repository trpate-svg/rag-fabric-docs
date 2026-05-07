import os
import PyPDF2
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Charge les variables d'environnement depuis .env

load_dotenv()

# --- Initialisation des clients ---
# ChromaDB : notre base de données vectorielle, stockée localement dans ./chroma_db

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Fonction d'embedding locale avec sentence-transformers
# Tourne directement sur ton Mac, sans API, sans limite, sans coût

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # modèle léger (~80MB), rapide et efficace
)

# Une "collection" = une table dans ChromaDB. On la crée si elle n'existe pas encore.

collection = chroma_client.get_or_create_collection(
    name="fabric_docs",
    embedding_function=embedding_fn
)


def extract_text_from_pdf(pdf_path):
    """
    Ouvre un PDF et concatène le texte de toutes ses pages en une seule string.
    PyPDF2 lit page par page — on ajoute un saut de ligne entre chaque page.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Découpe un texte long en petits morceaux (chunks) pour deux raisons :
    1. Les modèles d'embedding ont une limite de tokens en entrée
    2. Des chunks courts = des résultats de recherche plus précis

    chunk_size=500 : chaque chunk fait ~500 mots
    overlap=50     : les 50 derniers mots d'un chunk sont répétés au début du suivant,
                     pour ne pas couper un raisonnement à cheval sur deux chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_pdfs(docs_folder="./docs"):
    """
    Pipeline complet d'ingestion :
    1. Lire tous les PDFs du dossier docs/
    2. Extraire et chunker le texte
    3. Calculer les embeddings en local (sentence-transformers)
    4. Stocker chunks + embeddings dans ChromaDB
    """

    # Étape 1 — Lister les PDFs disponibles

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
    print(f"{len(pdf_files)} PDFs trouvés : {pdf_files}")

    # Ces trois listes seront passées ensemble à ChromaDB à la fin

    all_chunks = []                 # le texte brut de chaque chunk
    all_ids = []                    # un identifiant unique par chunk (requis par ChromaDB)
    all_metadatas = []              # infos supplémentaires : quel PDF, quel numéro de chunk

    # Étape 2 — Extraire et chunker chaque PDF

    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_folder, pdf_file)
        print(f"Traitement de {pdf_file}...")

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks générés")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{pdf_file}_chunk_{i}")                         # ex : "onelake.pdf_chunk_3"
            all_metadatas.append({"source": pdf_file, "chunk_index": i})    # traçabilité source

    # Étape 3+4 — ChromaDB calcule les embeddings et stocke en une seule opération
    # Pas d'API externe : sentence-transformers tourne localement sur ton Mac

    print(f"\nIndexation de {len(all_chunks)} chunks en local...")
    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
        # pas besoin de passer embeddings= : ChromaDB les calcule automatiquement
    )
    print(f"✅ {len(all_chunks)} chunks indexés avec succès !")


if __name__ == "__main__":
    ingest_pdfs()