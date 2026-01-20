from src.config import get_settings
from src.schemas import ArticleSchema
from src.connector import NeonArticlesSubject

import pathway as pw

from src.pipeline.build_docs import build_docs_table
from src.pipeline.rag_server import run_rag_server


def main():
    settings = get_settings()

    articles = pw.io.python.read(
        subject=NeonArticlesSubject(settings),
        schema=ArticleSchema,
    )

    docs = build_docs_table(articles)

    # IMPORTANT: this starts REST server + runs pipeline
    run_rag_server(docs, settings)


if __name__ == "__main__":
    main()
