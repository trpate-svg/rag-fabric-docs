import time
import query
import query_v2

# 10 questions de test — on couvre des cas variés :
# - questions larges (testent le sémantique)
# - questions avec termes exacts (testent le BM25)
# - questions croisées entre docs (testent la fusion)

QUESTIONS = [
    "Qu'est-ce que Microsoft Fabric ?",
    "Quelle est la différence entre Direct Lake et Import mode ?",
    "Comment fonctionne OneLake ?",
    "Quels sont les avantages de Direct Lake pour les grands volumes de données ?",
    "Qu'est-ce que Data Factory dans Microsoft Fabric ?",
    "Comment OneLake gère-t-il le stockage des données ?",
    "Quels sont les composants principaux de Microsoft Fabric ?",
    "Comment se connecter à OneLake depuis Power BI ?",
    "Quelles sont les limites du mode Import par rapport à Direct Lake ?",
    "Comment les workspaces fonctionnent-ils dans Fabric ?"
]


def run_benchmark():
    results = []

    print("=" * 70)
    print("BENCHMARK : RAG v1 (sémantique seul) vs RAG v2 (hybrid + rerank)")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}/10 : {question}")
        print(f"{'='*70}")

        # --- v1 ---
        print("\n[V1 — Sémantique seul]")
        t0 = time.time()
        answer_v1 = query.ask(question)
        time_v1 = round(time.time() - t0, 2)

        # --- v2 ---
        print("\n[V2 — Hybrid + Rerank]")
        t0 = time.time()
        answer_v2 = query_v2.ask(question)
        time_v2 = round(time.time() - t0, 2)

        results.append({
            "question": question,
            "time_v1": time_v1,
            "time_v2": time_v2,
            "answer_v1": answer_v1,
            "answer_v2": answer_v2
        })

    # --- Résumé ---
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 70)
    print(f"{'#':<4} {'Temps v1':>10} {'Temps v2':>10}  Question")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['time_v1']:>9}s {r['time_v2']:>9}s  {r['question'][:55]}")

    avg_v1 = round(sum(r["time_v1"] for r in results) / len(results), 2)
    avg_v2 = round(sum(r["time_v2"] for r in results) / len(results), 2)
    print("-" * 70)
    print(f"{'Moy':<4} {avg_v1:>9}s {avg_v2:>9}s")

    print("\n✅ Benchmark terminé. Résultats ci-dessus à copier dans le README.")
    return results


if __name__ == "__main__":
    run_benchmark()