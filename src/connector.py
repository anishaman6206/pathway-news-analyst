from __future__ import annotations

import time
from datetime import timezone
from typing import Dict, Any, List

import psycopg
from psycopg.rows import dict_row
import pathway as pw

from src.config import Settings


class NeonArticlesSubject(pw.io.python.ConnectorSubject):
    """
    Polls Neon Postgres for new rows in `articles` and emits them into Pathway.

    Assumptions:
      - Table: articles(id uuid, title text, content text, author text, source_name text, url text,
                      created_at timestamptz, published_at timestamptz, processed boolean)
      - Connector reads WHERE processed=false then marks processed=true after emit.
    """

    deletions_enabled = False

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def _fetch_unprocessed(self, conn) -> List[Dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  id,
                  title,
                  content,
                  COALESCE(author, '')      AS author,
                  COALESCE(source_name,'')  AS source_name,
                  COALESCE(url,'')          AS url,
                  created_at,
                  COALESCE(published_at, created_at) AS published_at
                FROM articles
                WHERE processed = false
                ORDER BY created_at ASC
                LIMIT %s;
                """,
                (self.settings.db_batch_size,),
            )
            return cur.fetchall()

    def _mark_processed(self, conn, ids: List[str]) -> None:
        if not ids:
            return
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET processed = true WHERE id = ANY(%s);",
                (ids,),
            )

    def run(self) -> None:
        while True:
            try:
                with psycopg.connect(self.settings.database_url, row_factory=dict_row) as conn:
                    rows = self._fetch_unprocessed(conn)

                    if not rows:
                        time.sleep(self.settings.db_poll_interval)
                        continue

                    emitted_ids: List[str] = []

                    for r in rows:
                        created_at = r.get("created_at")
                        published_at = r.get("published_at")

                        created_at_iso = (
                            created_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
                            if created_at
                            else ""
                        )
                        published_at_iso = (
                            published_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
                            if published_at
                            else created_at_iso
                        )

                        if self.settings.debug_emit:
                            print("[EMIT]", str(r["id"]), (r.get("title") or "")[:80])

                        self.next(
                            article_id=str(r["id"]),
                            title=r.get("title") or "",
                            content=r.get("content") or "",
                            author=r.get("author") or "",
                            source_name=r.get("source_name") or "manual",
                            url=r.get("url") or "",
                            created_at=created_at_iso,
                            published_at=published_at_iso,
                        )

                        emitted_ids.append(str(r["id"]))

                    self._mark_processed(conn, emitted_ids)
                    conn.commit()

            except Exception as e:
                print("[NeonArticlesSubject] error:", e)
                time.sleep(max(self.settings.db_poll_interval, 3))
