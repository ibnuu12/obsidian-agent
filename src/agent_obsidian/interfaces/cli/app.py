from __future__ import annotations

import argparse
from collections.abc import Sequence

from agent_obsidian.application.agent_models import RetrievalOptions
from agent_obsidian.application.indexing.vault_indexer import (
    DEFAULT_CONTENT_ROOTS,
)
from agent_obsidian.application.llm.exceptions import LLMError
from agent_obsidian.interfaces.cli.commands.ask import execute_ask
from agent_obsidian.interfaces.cli.commands.index import execute_index


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from exc

    if result < 1:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return result


def non_negative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from exc

    if result < 0:
        raise argparse.ArgumentTypeError(
            "value must be zero or greater"
        )

    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Local-first AI agent for Obsidian.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="agent-obsidian 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question about your Obsidian vault.",
    )

    ask_parser.add_argument(
        "query",
        help="Question to ask about your Obsidian vault.",
    )

    ask_parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum number of retrieved notes.",
    )

    ask_parser.add_argument(
        "--graph-limit",
        type=non_negative_int,
        default=5,
        help="Maximum number of graph-expanded notes.",
    )

    ask_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable knowledge graph expansion.",
    )

    ask_parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Filter by tag. Can be specified multiple times.",
    )

    ask_parser.add_argument(
        "--note-type",
        action="append",
        default=None,
        help="Filter by note type. Can be specified multiple times.",
    )

    ask_parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Filter by note status. Can be specified multiple times.",
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Index Markdown notes from your Obsidian vault.",
    )

    index_parser.add_argument(
        "--root",
        required=True,
        choices=DEFAULT_CONTENT_ROOTS,
        help="Vault domain to index.",
    )

    subparsers.add_parser(
        "status",
        help="Show agent configuration and status.",
    )

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "ask":
        retrieval = RetrievalOptions(
            limit=args.limit,
            expand_graph=not args.no_graph,
            graph_limit=args.graph_limit,
            tags=(
                frozenset(args.tag)
                if args.tag
                else None
            ),
            note_types=(
                frozenset(args.note_type)
                if args.note_type
                else None
            ),
            statuses=(
                frozenset(args.status)
                if args.status
                else None
            ),
        )

        try:
            return execute_ask(
                query=args.query,
                retrieval=retrieval,
            )
        except LLMError as exc:
            print(f"LLM error: {exc}")
            return 1
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1

    if args.command == "index":
        return execute_index(root=args.root)

    if args.command == "status":
        parser.error("The status command is not implemented yet.")

    return 0