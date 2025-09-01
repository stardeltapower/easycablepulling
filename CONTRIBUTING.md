# Contributing to Easy Cable Pulling

We welcome contributions to Easy Cable Pulling! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/easycablepulling.git
   cd easycablepulling
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure all tests pass:
   ```bash
   pytest
   ```

3. Run code quality checks:
   ```bash
   black .
   isort .
   flake8
   mypy easycablepulling
   ```

4. Commit your changes (pre-commit hooks will run automatically):
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

5. Push to your fork and create a pull request

## Code Style

- We use Black for code formatting (88 character line length)
- We use isort for import sorting
- All code must pass flake8 linting
- Type hints are required for all functions

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a PR
- Aim for high test coverage (>90%)
- Use pytest fixtures for test data

## Documentation

- Update docstrings for all public functions and classes
- Update README.md if adding new features
- Add examples for complex functionality

## Pull Request Process

1. Ensure your branch is up to date with the main branch
2. All tests must pass
3. Code must pass all quality checks
4. Update documentation as needed
5. Request review from maintainers

## Reporting Issues

- Use GitHub Issues to report bugs or request features
- Provide detailed information and steps to reproduce
- Include relevant code examples or error messages
