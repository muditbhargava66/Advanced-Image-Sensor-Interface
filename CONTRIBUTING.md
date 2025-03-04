# Contributing to Advanced Image Sensor Interface

Thank you for your interest in contributing to the Advanced Image Sensor Interface project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [muditbhargava666@gmail.com](mailto:muditbhargava666@gmail.com).

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```
   git clone https://github.com/yourusername/Advanced-Image-Sensor-Interface.git
   cd Advanced-Image-Sensor-Interface
   ```
3. **Set up the development environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -e ".[dev,docs]"
   ```
4. **Create a new branch** for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes in the appropriate files.
2. Run the automated tests and linting:
   ```
   tox
   ```
3. If you've added new functionality, add tests for it in the `tests/` directory.
4. Update documentation as needed in the `docs/` directory.
5. Commit your changes with a descriptive commit message:
   ```
   git commit -m "Add feature: your feature description"
   ```
6. Push your branch to your fork:
   ```
   git push origin feature/your-feature-name
   ```
7. Create a pull request against the main repository.

## Coding Standards

This project follows these coding standards:

- **PEP 8** for general Python coding style.
- **Black** for code formatting (line length of 130 characters).
- **MyPy** for type annotations.
- **Ruff** for additional linting.

These tools are configured in the `pyproject.toml` file and can be run using `tox` or directly.

### Type Annotations

All new code should include type annotations. Example:

```python
def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio.
    
    Args:
        signal: The clean signal or reference image.
        noise: The noise component.
        
    Returns:
        The calculated SNR in decibels.
    """
    # Implementation...
```

## Testing

This project uses `pytest` for testing. All new code should include appropriate tests.

### Running Tests

To run the tests:

```
pytest
```

For more detailed output:

```
pytest -v
```

To run tests with coverage:

```
pytest --cov=src
```

See the [Testing Guide](docs/testing_guide.md) for more detailed information on test design and implementation.

## Documentation

Documentation should be updated whenever you change code. This includes:

- **Docstrings** for all modules, classes, and functions.
- **API documentation** in `docs/api_documentation.md`.
- **Design specifications** in `docs/design_specs.md` if you're making architectural changes.
- **Performance analysis** in `docs/performance_analysis.md` if you're changing performance-critical code.

### Building Documentation

To build the documentation:

```
cd docs
make html
```

The built documentation will be available in `docs/_build/html/`.

## Pull Request Process

1. Ensure your code passes all tests and linting checks.
2. Update documentation as needed.
3. Add your change to the CHANGELOG.md file in the "Unreleased" section.
4. Make sure your PR description clearly describes the problem and solution. Include any relevant issue numbers.

### PR Review Checklist

Before submitting your PR, check that:

- [ ] All tests pass, including any new tests for your feature
- [ ] Code follows the project's coding standards
- [ ] Documentation is updated
- [ ] Changelog is updated

## Release Process

Releases are managed by the project maintainers. The process is as follows:

1. Update the version number in `pyproject.toml`.
2. Move entries from "Unreleased" to a new version section in CHANGELOG.md.
3. Create a new GitHub release with the version number as the tag.
4. Package and upload the release to PyPI.

## Questions or Need Help?

If you have questions or need help with the contribution process, please open an issue on GitHub or contact the project maintainers.

Thank you for contributing to the Advanced Image Sensor Interface project!