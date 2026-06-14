from typing import Any

from .sentiment_model import ABSASentimentModel, RuleBasedABSABaseline

__all__ = [
    "ABSASentimentModel",
    "RuleBasedABSABaseline",
    "BaselineModelInterface",
    "create_model_interface",
    "get_model_interface",
]


def __getattr__(name: str) -> Any:
    if name in {
        "BaselineModelInterface",
        "create_model_interface",
        "get_model_interface",
    }:
        from .interface import (
            BaselineModelInterface,
            create_model_interface,
            get_model_interface,
        )

        return {
            "BaselineModelInterface": BaselineModelInterface,
            "create_model_interface": create_model_interface,
            "get_model_interface": get_model_interface,
        }[name]
    raise AttributeError(name)
