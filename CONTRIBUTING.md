# Contributing to Wodel

Thank you for interest in contributing to the Wodel project! This guide will help you set up your development environment and get started with the codebase.

## Development Setup

We use [uv](https://astral.sh/uv/) as our fast Python package and environment manager.

### Prerequisites

If you don't have `uv` installed, you can install it using curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. Clone this repository to your local machine.
2. Initialize and synchronize the virtual environment and project dependencies:
   ```bash
   uv sync
   ```
   This command automatically creates a virtual environment in `.venv/` and installs all dependencies specified in `pyproject.toml`.

### Running Modules

Since the project is structured as flat modules at the root, always run files as Python modules from the project root using the `uv run python -m` format. This ensures correct path resolving:

* **Data Pipeline (DataLoader, DataCleaner, and FeatureBuilder)**:
  ```bash
  uv run python -m engine.data.Pipeline
  ```
* **Model Training & Evaluation (ModelFinder)**:
  ```bash
  uv run python -m engine.model.ModelFinder
  ```

### Managing Dependencies

Use `uv` commands to manage project requirements:
* **Add a package**: `uv add <package_name>`
* **Remove a package**: `uv remove <package_name>`
* **Update the lockfile and sync**: `uv sync`

### Code Quality (Linting & Formatting)

We use [Ruff](https://astral.sh/ruff/) to format and lint our code. Always check your changes before pushing:
* **Run Linter**: `uv run ruff check .`
* **Format Code**: `uv run ruff format .`

A GitHub Actions CI workflow runs these checks automatically on every push and pull request to `main`.

---