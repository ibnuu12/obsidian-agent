
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pathlib import PurePosixPath
from agent_obsidian.infrastructure.retrieval.note_parser import ParsedNote

SEARCH_STOPWORDS = {
    # English
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been",
    "what", "who", "when", "where", "why", "how",
    "do", "does", "did",
    "can", "could", "would", "should",
    "of", "to", "in", "on", "for", "with", "about",
    "and", "or", "but",
    "explain", "describe", "tell", "me",

    # Indonesian
    "apa", "apakah", "siapa", "kapan", "dimana",
    "mengapa", "kenapa", "bagaimana",
    "adalah", "ialah", "yang", "dan", "atau",
    "di", "ke", "dari", "pada", "untuk", "dengan",
    "tentang", "ini", "itu", "saya", "tolong",
    "jelaskan", "bisakah",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    note_id TEXT,
    title TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    note_type TEXT,
    status TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    frontmatter_json TEXT NOT NULL DEFAULT '{}',
    content TEXT NOT NULL,
    created TEXT,
    updated TEXT,
    mtime REAL,
    size INTEGER,
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notes_type ON notes(note_type);
CREATE INDEX IF NOT EXISTS idx_notes_status ON notes(status);
CREATE INDEX IF NOT EXISTS idx_notes_mtime ON notes(mtime);

CREATE TABLE IF NOT EXISTS note_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    target TEXT NOT NULL,
    target_path TEXT,
    relationship_type TEXT NOT NULL,
    UNIQUE(source_path, target, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_relationship_source
ON note_relationships(source_path);

CREATE INDEX IF NOT EXISTS idx_relationship_target
ON note_relationships(target_path);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title,
    aliases,
    tags,
    content,
    tokenize='unicode61'
);
"""


class SQLiteIndex:

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(SCHEMA)
            self._ensure_schema_migrations()
            self.connection.commit()

        except BaseException:
            # Preserve the original initialization error.
            # Cleanup errors must not hide the original failure.
            try:
                self.connection.close()
            except BaseException:
                pass

            raise


    def close(self):
        self.connection.close()

    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()

        except BaseException:
            # Always attempt cleanup while preserving
            # the original commit/rollback exception.
            try:
                self.close()
            except BaseException:
                pass

            raise

        else:
            self.close()


    @staticmethod
    def _normalize_reference(value: str) -> str:
        return (
            value.strip()
            .replace("\\", "/")
            .removeprefix("./")
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    def upsert_note(
        self,
        note: ParsedNote,
        mtime: float | None = None,
        size: int | None = None,
    ):
        with self.connection:
            existing = self.connection.execute(
                "SELECT id FROM notes WHERE path = ?",
                (note.path,),
            ).fetchone()

            if existing:
                row_id = existing["id"]

                self.connection.execute(
                    "DELETE FROM notes_fts WHERE rowid = ?",
                    (row_id,),
                )

                self.connection.execute(
                    """
                    UPDATE notes SET
                        note_id = ?,
                        title = ?,
                        aliases_json = ?,
                        note_type = ?,
                        status = ?,
                        tags_json = ?,
                        frontmatter_json = ?,
                        content = ?,
                        created = ?,
                        updated = ?,
                        mtime = ?,
                        size = ?,
                        indexed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        note.note_id,
                        note.title,
                        self._json(note.aliases),
                        note.note_type,
                        note.status,
                        self._json(note.tags),
                        self._json(note.frontmatter),
                        note.content,
                        note.created,
                        note.updated,
                        mtime,
                        size,
                        row_id,
                    ),
                )
            else:
                cursor = self.connection.execute(
                    """
                    INSERT INTO notes (
                        path, note_id, title, aliases_json,
                        note_type, status, tags_json,
                        frontmatter_json, content,
                        created, updated, mtime, size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note.path,
                        note.note_id,
                        note.title,
                        self._json(note.aliases),
                        note.note_type,
                        note.status,
                        self._json(note.tags),
                        self._json(note.frontmatter),
                        note.content,
                        note.created,
                        note.updated,
                        mtime,
                        size,
                    ),
                )
                row_id = cursor.lastrowid

            self.connection.execute(
                """
                INSERT INTO notes_fts (
                    rowid, title, aliases, tags, content
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    note.title,
                    " ".join(note.aliases),
                    " ".join(note.tags),
                    note.content,
                ),
            )

            self.connection.execute(
                "DELETE FROM note_relationships WHERE source_path = ?",
                (note.path,),
            )

            for target in note.wikilinks:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO note_relationships
                    (source_path, target, relationship_type)
                    VALUES (?, ?, 'wikilink')
                    """,
                    (note.path, target),
                )

            for target in note.related:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO note_relationships
                    (source_path, target, relationship_type)
                    VALUES (?, ?, 'related')
                    """,
                    (note.path, target),
                )

    def count_notes(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS total FROM notes"
        ).fetchone()
        return row["total"]

    def search(
        self,
        query: str,
        limit: int = 10,
        notes_prefix: str | None = None,
    ) -> list[dict]:
        import re

        raw_terms = re.findall(
            r"[\w]+",
            query,
            flags=re.UNICODE,
        )

        if not raw_terms:
            return []

        terms = [
            term
            for term in raw_terms
            if term.casefold() not in SEARCH_STOPWORDS
        ]

        # Preserve search behavior when the query contains
        # only stopwords.
        if not terms:
            terms = raw_terms

        fts_query = " AND ".join(
            '"' + term.replace('"', '""') + '"'
            for term in terms
        )

        sql = """
            SELECT
                n.path,
                n.title,
                n.note_type,
                n.status,
                n.tags_json,
                n.mtime,
                snippet(notes_fts, 3, '[', ']', ' ... ', 20) AS excerpt,
                bm25(notes_fts) AS rank
            FROM notes_fts
            JOIN notes n ON n.id = notes_fts.rowid
            WHERE notes_fts MATCH ?
        """

        parameters: list = [fts_query]

        if notes_prefix:
            sql += " AND n.path LIKE ?"
            parameters.append(notes_prefix.rstrip("/") + "/%")

        sql += " ORDER BY rank LIMIT ?"
        parameters.append(limit)

        rows = self.connection.execute(
            sql,
            parameters,
        ).fetchall()

        return [
            {
                "path": row["path"],
                "title": row["title"],
                "type": row["note_type"],
                "status": row["status"],
                "tags": json.loads(row["tags_json"]),
                "mtime": row["mtime"],
                "excerpt": row["excerpt"],
                "score": row["rank"],
            }
            for row in rows
        ]

    def get_note(self, path: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM notes WHERE path = ?",
            (path,),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["aliases"] = json.loads(result.pop("aliases_json"))
        result["tags"] = json.loads(result.pop("tags_json"))
        result["frontmatter"] = json.loads(
            result.pop("frontmatter_json")
        )

        return result

    def _ensure_schema_migrations(self):
        columns = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(note_relationships)"
            ).fetchall()
        }

        if "resolution_status" not in columns:
            self.connection.execute(
                """
                ALTER TABLE note_relationships
                ADD COLUMN resolution_status TEXT
                NOT NULL DEFAULT 'unresolved'
                """
            )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_relationship_status
            ON note_relationships(resolution_status)
            """
        )

        self.connection.commit()

    def rebuild_relationship_resolution(self) -> dict[str, int]:
        """
        Resolve all stored relationship targets against indexed notes.

        Must be called after notes have been indexed.
        """

        notes = self.connection.execute(
            """
            SELECT path, title, aliases_json
            FROM notes
            """
        ).fetchall()

        path_map = {}
        name_map = {}

        def add_candidate(mapping, key, path):
            key = key.casefold().strip()

            if key:
                mapping.setdefault(key, set()).add(path)

        for note in notes:
            path = note["path"]
            title = note["title"]

            aliases = json.loads(note["aliases_json"])
            note_path = PurePosixPath(path)

            add_candidate(path_map, path, path)
            add_candidate(path_map, path.removesuffix(".md"), path)

            add_candidate(name_map, title, path)
            add_candidate(name_map, note_path.stem, path)

            for alias in aliases:
                add_candidate(name_map, alias, path)

        relationships = self.connection.execute(
            """
            SELECT id, target
            FROM note_relationships
            """
        ).fetchall()

        counts = {
            "resolved": 0,
            "unresolved": 0,
            "ambiguous": 0,
        }

        with self.connection:
            for relationship in relationships:
                target = self._normalize_reference(
                    relationship["target"]
                )

                target_without_extension = target.removesuffix(".md")

                candidates = (
                    path_map.get(target.casefold())
                    or path_map.get(target_without_extension.casefold())
                    or name_map.get(target.casefold())
                )

                if not candidates:
                    status = "unresolved"
                    target_path = None

                elif len(candidates) > 1:
                    status = "ambiguous"
                    target_path = None

                else:
                    status = "resolved"
                    target_path = next(iter(candidates))

                self.connection.execute(
                    """
                    UPDATE note_relationships
                    SET target_path = ?,
                        resolution_status = ?
                    WHERE id = ?
                    """,
                    (
                        target_path,
                        status,
                        relationship["id"],
                    ),
                )

                counts[status] += 1

        return counts

    def get_outgoing_links(
        self,
        source_path: str,
        relationship_type: str | None = None,
    ) -> list[dict]:

        sql = """
            SELECT
                r.target,
                r.target_path,
                r.relationship_type,
                r.resolution_status,
                n.title AS target_title,
                n.note_type AS target_type
            FROM note_relationships r
            LEFT JOIN notes n ON n.path = r.target_path
            WHERE r.source_path = ?
        """

        parameters = [source_path]

        if relationship_type:
            sql += " AND r.relationship_type = ?"
            parameters.append(relationship_type)

        sql += " ORDER BY r.relationship_type, r.target"

        rows = self.connection.execute(
            sql, parameters
        ).fetchall()

        return [dict(row) for row in rows]

    def get_backlinks(self, target_path: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT
                r.source_path,
                r.relationship_type,
                n.title AS source_title,
                n.note_type AS source_type
            FROM note_relationships r
            JOIN notes n ON n.path = r.source_path
            WHERE r.target_path = ?
            ORDER BY n.title
            """,
            (target_path,),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_unresolved_links(
        self,
        status: str | None = None,
    ) -> list[dict]:

        sql = """
            SELECT
                source_path,
                target,
                relationship_type,
                resolution_status
            FROM note_relationships
            WHERE resolution_status IN (
                'unresolved',
                'ambiguous'
            )
        """

        parameters = []

        if status:
            if status not in ("unresolved", "ambiguous"):
                raise ValueError("Invalid resolution status")

            sql += " AND resolution_status = ?"
            parameters.append(status)

        sql += " ORDER BY source_path, target"

        rows = self.connection.execute(
            sql, parameters
        ).fetchall()

        return [dict(row) for row in rows]
