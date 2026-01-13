import os
from dotenv import load_dotenv
import pathway as pw

from src.connector import NeonArticlesSubject

load_dotenv()

class ArticleSchema(pw.Schema):
    id: str
    title: str
    content: str
    author: str
    source_name: str
    url: str
    created_at: str
    published_at: str

articles = pw.io.python.read(
    subject=NeonArticlesSubject(),
    schema=ArticleSchema,
)

# Debug: print incoming stream
pw.debug.compute_and_print(articles)

pw.run()
