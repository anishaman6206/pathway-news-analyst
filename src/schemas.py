import pathway as pw


class ArticleSchema(pw.Schema):
    article_id: str = pw.column_definition(primary_key=True)
    title: str
    content: str
    author: str
    source_name: str
    url: str
    created_at: str
    published_at: str
