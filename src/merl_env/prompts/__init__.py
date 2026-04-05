"""Prompt template package."""

from __future__ import annotations

from importlib import resources


def load_prompt_template(path: str) -> str:
    """Load a text prompt template from package resources."""

    resource = resources.files("merl_env.prompts").joinpath(path)
    return resource.read_text(encoding="utf-8")


__all__ = ["load_prompt_template"]
