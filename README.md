# RAG Fabric Docs

RAG (Retrieval Augmented Generation) sur la documentation Microsoft Fabric.
Pose des questions en langage naturel sur les PDFs Microsoft Learn — Claude génère une réponse basée uniquement sur les documents indexés.

## Stack

- **Embeddings** : sentence-transformers (`all-MiniLM-L6-v2`) — local, sans API
- **Vector store** : ChromaDB (persisté sur disque)
- **LLM** : Claude Sonnet (Anthropic API)
- **PDF parsing** : PyPDF2

## Architecture RAG

Question → Embedding → ChromaDB (recherche sémantique) → Top 5 chunks → Claude → Réponse

## Installation

1. uv install
2. Copie .env.example en .env et renseigne ta clé ANTHROPIC_API_KEY

## Utilisation

Étape 1 — Indexer les PDFs (à faire une seule fois) : uv run src/ingest.py
Étape 2 — Poser des questions : uv run src/query.py

## 5 questions d'exemple

| Question | Sources trouvées |
|----------|-----------------|
| Qu'est-ce que Microsoft Fabric ? | fabric-overview.pdf |
| Quelle est la différence entre Direct Lake et Import mode ? | direct-lake.pdf, fabric-overview.pdf |
| Comment fonctionne OneLake ? | onelake.pdf, fabric-overview.pdf |
| Quels sont les avantages de Direct Lake pour les grands volumes de données ? | direct-lake.pdf, fabric-overview.pdf |
| Qu'est-ce que Data Factory dans Microsoft Fabric ? | fabric-overview.pdf |

## Docs indexées

- `fabric-overview.pdf` — Vue d'ensemble Microsoft Fabric
- `onelake.pdf` — Architecture OneLake
- `direct-lake.pdf` — Mode Direct Lake vs Import