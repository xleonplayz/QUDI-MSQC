# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands
- Install: `pip install -e .`
- Run qudi: `qudi`
- Run single test: `jupyter-nbconvert --execute tests/notebooks/test_name.ipynb`
- Run all tests: `bash tests/test.sh`

## Code Style Guidelines
- Use PEP 8 style for Python code
- Include license header in all new files
- Type annotations with `typing` module for function/method signatures
- Document classes and methods with docstring comments
- Use abstract methods for interfaces with `@abstractmethod`
- Implement hardware modules with `OverloadedAttribute` when implementing multiple interfaces
- Error handling: wrap with try/except and log errors with `self.log.exception()`
- Format imports: standard libraries first, then third-party, then qudi modules
- Fit models should inherit from `FitModelBase` and implement required methods
- Use StatusVar for persistent module variables
- Use ConfigOption for module configuration parameters
- Document Qt signal names with pattern `sig*Changed`