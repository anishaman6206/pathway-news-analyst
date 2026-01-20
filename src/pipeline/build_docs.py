import pathway as pw


@pw.udf
def to_bytes(s: str) -> bytes:
    return s.encode("utf-8", errors="ignore")


def build_docs_table(articles: pw.Table) -> pw.Table:
    """
    Convert articles table -> docs table (bytes payload + metadata)
    """
    articles = articles.with_columns(
        doc_text=pw.this.title + "\n\n" + pw.this.content
    )

    docs = articles.select(
        data=to_bytes(pw.this.doc_text),
        article_id=pw.this.article_id,
        title=pw.this.title,
        source_name=pw.this.source_name,
        url=pw.this.url,
        published_at=pw.this.published_at,
    )
    return docs


def article_parser(raw: bytes):
    """
    Parser required by DocumentStore: bytes -> list[(text, metadata)].
    """
    text = raw.decode("utf-8", errors="ignore")
    return [(text, {})]
