
import sqlite3

from agent_obsidian.infrastructure.retrieval.sqlite_index import SQLiteIndex


def test_migrate_legacy_relationship_schema(tmp_path):
    database_path = tmp_path / "legacy.db"

    # Simulasikan database versi lama tanpa resolution_status.
    legacy_connection = sqlite3.connect(database_path)

    legacy_connection.execute(
        """
        CREATE TABLE note_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL,
            target TEXT NOT NULL,
            target_path TEXT,
            relationship_type TEXT NOT NULL,
            UNIQUE(source_path, target, relationship_type)
        )
        """
    )

    legacy_connection.execute(
        """
        INSERT INTO note_relationships (
            source_path,
            target,
            target_path,
            relationship_type
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            "Notes/Source.md",
            "Target Note",
            "Notes/Target Note.md",
            "wikilink",
        ),
    )

    legacy_connection.commit()
    legacy_connection.close()

    # Buka database lama menggunakan implementasi terbaru.
    index = SQLiteIndex(database_path)

    try:
        columns = {
            row["name"]
            for row in index.connection.execute(
                "PRAGMA table_info(note_relationships)"
            ).fetchall()
        }

        assert "resolution_status" in columns

        relationship = index.connection.execute(
            """
            SELECT *
            FROM note_relationships
            WHERE source_path = ?
            """,
            ("Notes/Source.md",),
        ).fetchone()

        # Data lama harus tetap tersimpan.
        assert relationship is not None
        assert relationship["target"] == "Target Note"
        assert relationship["relationship_type"] == "wikilink"

        # Baris lama mendapat nilai default dari migrasi.
        assert relationship["resolution_status"] == "unresolved"

        # Pastikan index berhasil dibuat.
        indexes = {
            row["name"]
            for row in index.connection.execute(
                "PRAGMA index_list(note_relationships)"
            ).fetchall()
        }

        assert "idx_relationship_status" in indexes

    finally:
        index.connection.close()

    # Pastikan migrasi aman dijalankan kembali.
    reopened_index = SQLiteIndex(database_path)

    try:
        count = reopened_index.connection.execute(
            "SELECT COUNT(*) FROM note_relationships"
        ).fetchone()[0]

        assert count == 1
    finally:
        reopened_index.connection.close()
