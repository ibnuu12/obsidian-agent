
import json


def print_mcp_result(result):
    structured = getattr(result, "structured_content", None)

    if structured is not None:
        print(json.dumps(
            structured,
            indent=2,
            ensure_ascii=False,
            default=str,
        ))
        return

    for item in result.content:
        if hasattr(item, "text"):
            print(item.text)
        else:
            print(repr(item))
