import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.

    :param text: The text to count tokens for
    :param model: The model name to use for token counting (default: gpt-4o)
    :return: Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))