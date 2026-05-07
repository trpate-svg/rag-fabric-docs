# RAG Fabric Docs

## Objectif
RAG (Retrieval Augmented Generation) sur la documentation Microsoft Fabric.
Permet de poser des questions en langage naturel sur les PDFs Microsoft Fabric.

## Stack
- Python 3.12 / uv
- Embeddings : Voyage AI (voyage-3)
- Vector store : ChromaDB (local)
- LLM : Claude (claude-sonnet-4-20250514) via API Anthropic
- PDF parsing : PyPDF2

## Structure
- /docs        → PDFs Microsoft Fabric téléchargés
- /src         → code source
- chroma_db/   → base vectorielle locale (ne pas commiter)

## Conventions
- Variables d'environnement dans .env (jamais hardcodées)
- Un script par étape : ingest.py, query.py
- Commits après chaque étape fonctionnelle