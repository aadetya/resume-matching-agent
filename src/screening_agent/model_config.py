"""Active model default and endpoint-compatible generation settings."""

DEFAULT_MODEL = "gpt-5.6-luna"


def generation_options(model: str) -> dict:
    # Preserve the previous non-reasoning latency contract. GPT-5.6 otherwise
    # defaults to medium reasoning, sharing the output budget with hidden tokens.
    if model.startswith("gpt-5.6"):
        return {"reasoning": {"effort": "none"}}
    if model.startswith(("gpt-4.1", "gpt-4o")):
        return {"temperature": 0}
    return {}
