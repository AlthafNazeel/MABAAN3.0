"""Helper to build notebook JSON programmatically."""
import json


def _split_source(source):
    """Split source into lines, preserving \\n at end of each line (nbformat spec)."""
    lines = source.split("\n")
    # Every line except the last should end with \n
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source)}


def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "source": _split_source(source),
            "execution_count": None, "outputs": []}


def make_notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [{"sourceId": 14757019, "sourceType": "datasetVersion", "datasetId": 9432119}],
                "dockerImageVersionId": 31259,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True,
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved: {path}")
