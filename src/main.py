import sys
import argparse
import json
from src.file_loader import ParsingFileError, load_fn_definitions, load_prompts
from src.write_output import write_output, WritingOutputError
from src.basemodels import FunctionDefinition, PromptItem, FunctionCallResult
from src.constrained_decoding import decode_output, DecodingError
from typing import Any
from pydantic import ValidationError
import time
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"
BOLD = "\033[1m"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural-language prompts."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions."
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing input prompts."
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the JSON file where results will be written."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limits the number of tokens to this number"
    )

    return parser.parse_args()


def call_me_maybe() -> None:
    """
    Main function to process prompts and generate function call results.
    """
    args = parse_arguments()
    fn_def_path: str = args.functions_definition
    prompts_path: str = args.input
    output_path: str = args.output
    try:
        functions: list[FunctionDefinition] = load_fn_definitions(fn_def_path)
        prompts: list[PromptItem] = load_prompts(prompts_path)
        output: list[FunctionCallResult] = []
        model: Small_LLM_Model = Small_LLM_Model()
        vocab_path: str = model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            vocab_file = json.load(f)
            vocab: dict[int, str] = {
                v: k for k, v in vocab_file.items()
                }
        init_time: float = time.perf_counter()
        for prompt in prompts:
            print(f"{COLOR_GREEN}{BOLD}"
                  "\n============ PROCESSING... ============"
                  f"\n\n{COLOR_RESET}{prompt}")
            init_time_prompt: float = time.perf_counter()
            object_dict: dict[str, Any] = decode_output(
                prompt.prompt, functions, model, vocab
                )
            object: FunctionCallResult = FunctionCallResult(**object_dict)
            output.append(object)
            end_time_prompt: float = time.perf_counter()
            final_prompt_time: float = end_time_prompt - init_time_prompt
            print(f"\n{BOLD}{COLOR_YELLOW}Processed in: "
                  f"{final_prompt_time:.2f} seconds{COLOR_RESET}\n")
        output_dicts: Any = [
            result.model_dump() for result in output
            ]
        write_output(output_dicts, output_path)
        final_program_time: float = time.perf_counter()
        final_duration = final_program_time - init_time
        print(f"{BOLD}{COLOR_YELLOW}Total time processing all prompts: "
              f"{final_duration:.2f} seconds{COLOR_RESET}")
    except ImportError:
        print(
            "llm_sdk module not found. "
            "Please ensure it is installed and accessible.",
            file=sys.stderr
            )
        sys.exit(1)
    except (
                FileNotFoundError, PermissionError
            ) as e:
        print(
            f"An error ocurred while accessing the file. {e}",
            file=sys.stderr
        )
        sys.exit(1)
    except DecodingError as e:
        print(
            f"An error ocurred while decoding, {e}",
            file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"An error ocurred while validating an object, {e}",
              file=sys.stderr)
        sys.exit(1)
    except ParsingFileError as e:
        print(
            f"An error ocurred while parsing json files: {e}",
            file=sys.stderr)
        sys.exit(1)
    except WritingOutputError as e:
        print(
            f"An error ocurred while writing the output file: {e}",
            file=sys.stderr)
        sys.exit(1)
