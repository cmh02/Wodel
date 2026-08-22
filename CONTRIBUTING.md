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

---

## Coding Standards

Please adhere to the coding guidelines defined in [STANDARDS.md](file:///.agents/STANDARDS.md):
- **Naming Style**: Use `camelCase` for variable names, method names, and database column designations (e.g. `buildFeatures`, `timeSinceLastWorkout`).
- **Layout & Headers**: Include the module header docstring (with Title and Author fields) and structured Google-style docstrings for classes, methods, and functions.
