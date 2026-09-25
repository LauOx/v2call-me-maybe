import json
from typing import Any
from pathlib import Path


class WritingOutputError(Exception):
    """Custom error for writing output function"""
    pass


def write_output(output_dicts: dict[str, Any], output_path: str) -> None:
    """
    Write the output dictionaries to a JSON file.

    Args:
        output_dicts: A list of dictionaries representing the output.
        path: The path to the output file.

    Raises:
        WritingOutputError: If there is an error writing the output file.
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as output_file:
            output_json = json.dumps(output_dicts, indent=2)
            output_file.write(output_json)
    except (FileNotFoundError, IsADirectoryError):
        raise WritingOutputError(
            f"'{path}' does not point to a valid folder"
            ) from None
    except PermissionError:
        raise WritingOutputError(
            f"'{path}' Permission denied"
            ) from None
    except OSError:
        raise WritingOutputError(
            f"unexpected OS error while writing {path}"
            ) from None
