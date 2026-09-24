from src.enterprise_rag.config import load_config
from src.enterprise_rag.evaluation.runner import evaluate_bm25, load_dataset
from src.enterprise_rag.rag_system import RagSystem


def main() -> None:
    config = load_config()
    dataset = load_dataset(config.paths.evaluation_dataset)
    system = RagSystem(config=config, bm25_only=True)
    summary = evaluate_bm25(system, dataset, top_k=config.retrieval.top_k)
    print(summary)


if __name__ == "__main__":
    main()


