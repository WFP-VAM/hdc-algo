# hdc-algo

[![hdc-algo](https://github.com/WFP-VAM/hdc-algo/actions/workflows/main.yaml/badge.svg?branch=main)](https://github.com/WFP-VAM/hdc-algo/actions/workflows/main.yaml)

This repo contains (mostly numba accelerated) algorithmic code and selected additional functions used in the WFP HumanitarianDataCube (HDC).

## Installation

```bash
# mamba/conda
mamba install -c wfp-ram hdc-algo

# pip
pip install --extra-index-url=https://data.earthobservation.vam.wfp.org/pypi/ hdc-algo
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/WFP-VAM/hdc-algo.git
cd hdc-algo
uv sync --all-extras --dev

# Run tests
uv run pytest

# Run linting and formatting
uv run ruff check hdc tests
uv run ruff format hdc tests
uv run mypy hdc
```
