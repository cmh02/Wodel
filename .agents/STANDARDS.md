# Python Coding & Layout Guidelines

## 1. Module Header Layout
Every Python file (module) must start with a descriptive docstring at the top of the file containing the following structure:
```python
"""
Wodle - <Component/Module Name>
Author: FirstName LastName (@GitHubTag)

<Description of the module's responsibilities, API, and core concepts.>
"""
```

## 2. Class Docstrings
Classes should include a concise docstring layout directly below the class declaration:
```python
class MyClass:
    """
    Wodel ClassName

    A brief description of what the class initializes or its purpose in the service.
    """
```

## 3. Function and Method Docstrings
Functions and methods must include a formatted docstring following this structure:
- **Title**: A short, clear title indicating the action.
- **Description**: A paragraph explaining the function's purpose/behavior.
- **Args**: An indented arguments section listing each argument without duplicating type annotations (rely on type hints in the signature).
- **Returns**: A description of the return type and values.

```python
def my_function(self, param_name: str) -> bool:
    """
    Function Title - Specific Action

    A detailed description of the function helper, what format it expects,
    and any details on how it handles operations.

    Args:
        param_name: Description of the parameter.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
```

## 4. File Organization and Structure

This project is a "flat-repo" such that each component is kept at the top-level. There are currently two main components:

1. **`engine/`**: Contains all data preparation, cleaning, feature engineering, and modeling logic.
   - **`engine/data/`**: Handles loading, filtering, and engineering steps (e.g., `DataLoader.py`, `DataCleaner.py`, `FeatureBuilder.py`, `Pipeline.py`).
   - **`engine/model/`**: Handles model training and validation (e.g., `ModelFinder.py`).
2. **`frontend/`**: Reserved for the user interface, plotting, and interactive visualization code.

Additionally, data files and datasets should be placed in the top-level **`data/`** directory. All logs should be stored in the **`logs/`** directory. All AI-development-related files should be stored in the **`.agents/`** directory.

## 5. Naming Standards

All code and file components in the project must adhere to the following naming conventions:

- **File Names & Class Names**:
  - Must match exactly (e.g., the class `ModelFinder` resides in the file `ModelFinder.py`).
  - Must be written in **`ProperCase`** (PascalCase) (e.g., `DataCleaner.py`).
  - Must be descriptive of the module/logic contained within the file.
- **Function & Method Names**:
  - Must be written in **`camelCase`** (e.g., `loadFromStrongCSV`, `removeAnyNaN`).
- **Variable Names**:
  - Must be written in **`camelCase`** (e.g., `engineeredDf`, `finalCount`).

## 6. Code Quality & Standards (Ruff)

- We use **Ruff** to enforce formatting and code quality checks.
- Before completing any code changes, always format the code and run the linter:
  - **Check / Lint**: `uv run ruff check .`
  - **Format**: `uv run ruff format .`
- A GitHub Actions CI workflow runs these checks automatically on every push or pull request to the `main` branch. All code is expected to pass both checks before merge.
