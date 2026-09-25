from __future__ import annotations
from src.basemodels import ParameterSpec, FunctionDefinition
from torch import Tensor
from typing import Any
from src import aux
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"


class DecodingError(Exception):
    """Custom error for decoding"""
    pass


def decode_fn_name(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        allowed: list[str]) -> str:
    """
    Decode function name from the model
    Args:
        model: Small_LLM_Model: the model to use for decoding
        vocab: dict[int, str]: the vocabulary to use for decoding
        context: str: the context to use for decoding
        allowed: list[str]: the list of allowed function names
    Returns:
        str: the decoded function name
    Raises:
        DecodingError: if no matches found for function name
    """
    encoded: Tensor = model.encode(context)
    input_ids: list[int] = encoded[0].tolist()
    candidate: str = ""
    fn_name: str = ""
    param_value: str = ""
    status: aux.CandidateStatus = aux.check_candidate_logits(
        candidate, allowed
        )

    while status != aux.CandidateStatus.VALID_COMPLETE:
        logits: list[float] = model.get_logits_from_input_ids(input_ids)

        # invalid tokens to logit -inf
        for token_id in range(len(logits)):
            token_text = vocab.get(token_id)
            if token_text is None:
                logits[token_id] = float("-inf")
                continue
            if param_value != "":
                if param_value[-1] == 'Ġ' and token_text[0] == 'Ġ':
                    logits[token_id] = float("-inf")
            candidate = fn_name + token_text
            status = aux.check_candidate_logits(candidate, allowed)
            if status == aux.CandidateStatus.INVALID:
                logits[token_id] = float("-inf")
        # Check error
        if max(logits) == float("-inf"):
            raise DecodingError("No matches found for function name")

        # Add new token to answer
        next_token_id: int = logits.index(max(logits))
        input_ids.append(next_token_id)
        decoded_next_token: str = vocab[next_token_id]
        fn_name += decoded_next_token
        print(f"Function name found so far: '{fn_name}'")
        status = aux.check_candidate_logits(fn_name, allowed)

    return fn_name


def decode_param_number(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        type: str) -> str:
    """
    Decode function parameter if type is a kind of number
    Args:
        model: Small_LLM_Model: the model to use for decoding
        vocab: dict[int, str]: the vocabulary to use for decoding
        context: str: the context to use for decoding
        type: str: the type of the parameter to decode
    Returns:
        str: the decoded parameter value
    Raises:
        DecodingError: if no matches found for function parameter
    """
    context += ' "'
    encoded: Tensor = model.encode(context)
    input_ids: list[int] = encoded[0].tolist()
    candidate: str = ""
    param_value: str = ""
    status: bool = True
    while status:
        logits: list[float] = model.get_logits_from_input_ids(input_ids)

        # invalid tokens to logit -inf
        for token_id in range(len(logits)):
            token_text = vocab.get(token_id)
            if token_text is None:
                logits[token_id] = float("-inf")
                continue
            candidate = param_value + token_text

        # Check error
        if max(logits) == float("-inf"):
            raise DecodingError(
                "No matches found for function parameter"
                )

        # Add next token to answer
        print("Value found so far:", repr(param_value))
        next_token_id: int = logits.index(max(logits))
        input_ids.append(next_token_id)
        decoded_next_token: str = vocab[next_token_id]
        candidate = param_value + decoded_next_token
        # check stop char '"'
        if '"' in candidate:
            status = False
        if status:
            param_value += decoded_next_token
    return param_value


def decode_param_string(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        ) -> str:
    """
    Decode parameter if is a string
    Args:
        model: Small_LLM_Model: the model to use for decoding
        vocab: dict[int, str]: the vocabulary to use for decoding
        context: str: the context to use for decoding
    Returns:
        str: the decoded parameter value
    Raises:
        DecodingError: if no matches found for function parameter
    """
    context += '"'
    encoded: Tensor = model.encode(context)
    input_ids: list[int] = encoded[0].tolist()
    param_value: str = ""
    status = True
    while status:
        logits: list[float] = model.get_logits_from_input_ids(input_ids)

        # invalid tokens to logit -inf
        for token_id in range(len(logits)):
            token_text = vocab.get(token_id)
            if token_text is None:
                logits[token_id] = float("-inf")
                continue
            # Prevent double spaces
            if param_value != "":
                if param_value[-1] == 'Ġ' and token_text[0] == 'Ġ':
                    logits[token_id] = float("-inf")

        # Check error
        if param_value == "":
            if max(logits) == float("-inf"):
                raise DecodingError(
                    "No matches found for function parameter"
                    )

        next_token_id: int = logits.index(max(logits))
        input_ids.append(next_token_id)
        decoded_next_token: str = vocab[next_token_id]
        safe_part = aux.find_stop_char(decoded_next_token)
        if safe_part is not None:
            param_value += safe_part
            status = False
        else:
            param_value += decoded_next_token
            print(f"Value found so far: '{param_value}'")
            if param_value.endswith('Ċ'):
                status = False
                break
    return param_value


def decode_param_bool(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        ) -> str:
    """
    Decode parameter if is a boolean
    Args:
        model: Small_LLM_Model: the model to use for decoding
        vocab: dict[int, str]: the vocabulary to use for decoding
        context: str: the context to use for decoding
    Returns:
        str: the decoded parameter value
    Raises:
        DecodingError: if no matches found for function parameter
    """
    context += (
        "Based on the request, should 'strict' be true or false? Answer: "
    )
    encoded: Tensor = model.encode(context)
    input_ids: list[int] = encoded[0].tolist()
    candidate: str = ""
    param_value: str = ""
    status = True
    while status:
        logits: list[float] = model.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            token_text = vocab.get(token_id)
            if token_text is None:
                logits[token_id] = float("-inf")
                continue
            candidate = param_value + token_text
            if not ('true'.startswith(candidate) or
                    'false'.startswith(candidate)):
                logits[token_id] = float("-inf")

        if max(logits) == float("-inf"):
            raise DecodingError(
                "No matches found for function parameter"
                )
        next_token_id: int = logits.index(max(logits))
        input_ids.append(next_token_id)
        decoded_next_token: str = vocab[next_token_id]
        candidate = param_value + decoded_next_token
        status = not (candidate == 'true' or candidate == 'false')
        param_value += decoded_next_token
    return param_value


def decode_parameter(
        model: Small_LLM_Model,
        vocab: dict[int, str],
        context: str,
        type: str) -> str:
    """
    Decode a parameter based on its type
    Args:
        model: Small_LLM_Model: the model to use for decoding
        vocab: dict[int, str]: the vocabulary to use for decoding
        context: str: the context to use for decoding
        type: str: the type of the parameter
    Returns:
        str: the decoded parameter value
    """
    param_value: str = ""
    if (
        type == aux.ParamType.NUMBER.name or
        type == aux.ParamType.FLOAT.name or
        type == aux.ParamType.INTEGER.name
    ):
        param_value = decode_param_number(
            model, vocab, context, type
        )
    if type == aux.ParamType.STRING.name:
        param_value = decode_param_string(
            model, vocab, context
        )
    if type == aux.ParamType.BOOL.name:
        param_value = decode_param_bool(model, vocab, context)

    final_param_value: str = param_value.replace('Ġ', ' ')
    return final_param_value.rstrip().rstrip('Ċ')


def decode_output(
        prompt: str,
        functions: list[FunctionDefinition],
        model: Small_LLM_Model,
        vocab: dict[int, str],) -> dict[str, Any]:
    """
    Decode the output of the model into a dictionary of function calls
    Args:
        prompt: str: the prompt to use for decoding
        functions: list[FunctionDefinition]: the list of function definitions
        model: Small_LLM_Model: the model to use for decoding
        vocab: dict[int, str]: the vocabulary to use for decoding
    Returns:
        dict[str, Any]: the decoded function calls
    """
    object_return: dict[str, Any] = {}
    fn_name: str = ""

    # Find function name
    allowed_fn_names: list[str] = []
    for fn in functions:
        allowed_fn_names.append(fn.name)
    if len(allowed_fn_names) == 0:
        raise DecodingError(
            "No function names found in the function definitions input file"
            )
    fn_initial_context: str = aux.build_context_for_fn_name(functions, prompt)
    print(f"\n{COLOR_CYAN}Finding function name... "
          f"{COLOR_RESET}")
    fn_name = decode_fn_name(
        model, vocab, fn_initial_context, allowed_fn_names
        )

    # Find function parameters
    function: FunctionDefinition = next(
        fn for fn in functions
        if fn.name == fn_name
    )
    funct_param: dict[str, ParameterSpec] = function.parameters
    result_parameters: dict[str, Any] = {}
    param_initial_context: str = (
        aux.build_context_for_parameter(prompt, function)
    )
    param_value: str = ""
    for param in funct_param:
        print(f"\n{COLOR_CYAN}Finding parameter '{param}'...{COLOR_RESET}")
        type = aux.ParamType(funct_param[param].type).name
        param_initial_context += (
            f'"{param}":'
        )
        param_value = decode_parameter(
            model, vocab, param_initial_context, type
        )
        if (
            type == aux.ParamType.FLOAT.name or
            type == aux.ParamType.NUMBER.name or
            type == aux.ParamType.INTEGER.name
        ):
            if '.' in param_value:
                number_str: str = aux.clean_number(param_value)
                param_value = number_str
            if type == aux.ParamType.INTEGER.name:
                number = int(float(param_value))
                result_parameters[param] = number
            if (
                type == aux.ParamType.FLOAT.name or
                type == aux.ParamType.NUMBER.name
            ):
                result_parameters[param] = float(param_value)
        else:
            cleaned_param_value: str = aux.clean_param_value(param_value)
            final_param = aux.preserve_literal_string_value(
                cleaned_param_value, prompt
                )
            result_parameters[param] = final_param
        # Add param found for context to find next param
        param_initial_context += param_value + ', '
    # Save final dict result
    object_return = {
        'prompt': prompt,
        'name': fn_name,
        'parameters': result_parameters
    }
    print(f"{COLOR_CYAN}\nFinal object: \n{COLOR_RESET}{object_return}")
    return object_return
