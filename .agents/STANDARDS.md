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
