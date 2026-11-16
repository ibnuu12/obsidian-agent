
from __future__ import annotations

from typing import Any

from agent_obsidian.application.retrieval_protocol import RetrievalIndex

class HybridRetrievalService:
    """
    Combines FTS5 full-text search with Knowledge Graph expansion.

    Direct FTS results are prioritized. Resolved outgoing links and
    backlinks are added as graph context.
    """

    def __init__(self, index: RetrievalIndex):
        self.index = index

    @staticmethod
    def _matches_metadata(
        note: dict[str, Any],
        note_types: set[str] | None = None,
        statuses: set[str] | None = None,
        tags: set[str] | None = None,
    ) -> bool:
        note_type = note.get("type", note.get("note_type"))

        if note_types and note_type not in note_types:
            return False

        if statuses and note.get("status") not in statuses:
            return False

        if tags:
            note_tags = set(note.get("tags", []))

            if not note_tags.intersection(tags):
                return False

        return True

    def _enrich_note(
        self,
        path: str,
        source: str,
        excerpt: str | None = None,
        score: float | None = None,
        relationships: dict[str, list] | None = None,
    ) -> dict[str, Any] | None:
        note = self.index.get_note(path)

        if note is None:
            return None

        return {
            "path": note["path"],
            "title": note["title"],
            "type": note["note_type"],
            "status": note["status"],
            "tags": note["tags"],
            "aliases": note["aliases"],
            "frontmatter": note["frontmatter"],
            "content": note["content"],
            "excerpt": excerpt if excerpt is not None
            else note["content"][:500],
            "score": score,
            "source": source,
            "relationships": relationships or {
                "outgoing": [],
                "backlinks": [],
            },
        }

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        note_types: set[str] | None = None,
        statuses: set[str] | None = None,
        tags: set[str] | None = None,
        expand_graph: bool = True,
        graph_limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve notes using FTS5 and optionally expand through the graph.

        Args:
            query: User search query.
            limit: Maximum number of direct FTS results.
            note_types: Optional exact note type filters.
            statuses: Optional exact status filters.
            tags: Optional tags; at least one must match.
            expand_graph: Include resolved graph neighbors.
            graph_limit: Maximum number of additional graph notes.

        Returns:
            Direct results followed by graph context results.
        """
        if not query or not query.strip():
            return []

        if limit < 1:
            raise ValueError("limit must be greater than zero")

        if graph_limit < 0:
            raise ValueError("graph_limit cannot be negative")

        note_types = set(note_types) if note_types else None
        statuses = set(statuses) if statuses else None
        tags = set(tags) if tags else None

        # Retrieve extra candidates because metadata filtering happens
        # after the FTS5 query.
        candidate_limit = max(limit * 5, limit)

        candidates = self.index.search(
            query=query,
            limit=candidate_limit,
        )

        direct_results = []
        direct_paths = set()

        for candidate in candidates:
            path = candidate["path"]

            note = self.index.get_note(path)

            if note is None:
                continue

            metadata = {
                "type": note["note_type"],
                "status": note["status"],
                "tags": note["tags"],
            }

            if not self._matches_metadata(
                metadata,
                note_types=note_types,
                statuses=statuses,
                tags=tags,
            ):
                continue

            result = self._enrich_note(
                path=path,
                source="fts",
                excerpt=candidate.get("excerpt"),
                score=candidate.get("score"),
            )

            if result is None:
                continue

            direct_results.append(result)
            direct_paths.add(path)

            if len(direct_results) >= limit:
                break

        if not expand_graph or graph_limit == 0:
            return direct_results

        graph_results = []
        graph_paths = set(direct_paths)

        for direct in direct_results:
            source_path = direct["path"]

            outgoing = self.index.get_outgoing_links(source_path)
            backlinks = self.index.get_backlinks(source_path)

            direct["relationships"] = {
                "outgoing": outgoing,
                "backlinks": backlinks,
            }

            # Only resolved outgoing links can be expanded.
            for relationship in outgoing:
                target_path = relationship.get("target_path")

                if (
                    relationship.get("resolution_status") != "resolved"
                    or not target_path
                    or target_path in graph_paths
                ):
                    continue

                related_note = self.index.get_note(target_path)

                if related_note is None:
                    continue

                if not self._matches_metadata(
                    related_note,
                    note_types=note_types,
                    statuses=statuses,
                    tags=tags,
                ):
                    continue

                result = self._enrich_note(
                    path=target_path,
                    source="graph",
                )

                if result is None:
                    continue

                graph_results.append(result)
                graph_paths.add(target_path)

                if len(graph_results) >= graph_limit:
                    break

            if len(graph_results) >= graph_limit:
                break

            # Backlinks represent notes that point to this direct result.
            for relationship in backlinks:
                source_note_path = relationship["source_path"]

                if source_note_path in graph_paths:
                    continue

                related_note = self.index.get_note(source_note_path)

                if related_note is None:
                    continue

                if not self._matches_metadata(
                    related_note,
                    note_types=note_types,
                    statuses=statuses,
                    tags=tags,
                ):
                    continue

                result = self._enrich_note(
                    path=source_note_path,
                    source="graph",
                )

                if result is None:
                    continue

                graph_results.append(result)
                graph_paths.add(source_note_path)

                if len(graph_results) >= graph_limit:
                    break

            if len(graph_results) >= graph_limit:
                break

        return direct_results + graph_results
