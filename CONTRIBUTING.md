# Contributing to SyntheticDigits

Thank you for considering contributing to SyntheticDigits! We welcome contributions from everyone, whether it's a bug report, feature request, or code contribution.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

## How Can I Contribute?

### Reporting Bugs
- Ensure the bug was not already reported by searching the [Issues](https://github.com/BLShaw/SyntheticDigits/issues) page.
- If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Features
- First, check the [Issues](https://github.com/BLShaw/SyntheticDigits/issues) to see if the feature has already been suggested.
- Open a new issue if your feature idea is new. Provide a detailed explanation of the proposed feature and its use cases.

### Pull Requests
- Fork the repository and create your branch from `main`.
- If you've added code that should be tested, add tests.
- Ensure all tests pass (`python -m pytest tests/`).
- Make sure your code follows our style guidelines.
- Issue the pull request!

## Development Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/SyntheticDigits.git
   cd SyntheticDigits
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Make your changes and test them.

## Pull Request Process

1. Update the README.md with details of changes if applicable.
2. Increase the version number in any examples files and the README.md to reflect the changes.
3. Ensure all tests pass before submitting your pull request.
4. Your pull request will be reviewed and merged once approved.

## Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding standards.
- Write clear, descriptive commit messages.
- Use docstrings for all public functions and classes.
- Add type hints where possible.

Thank you again for your interest in contributing to SyntheticDigits!