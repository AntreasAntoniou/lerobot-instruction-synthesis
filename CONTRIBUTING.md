# Contributing to LeRobot Instruction Synthesis

Thank you for your interest in contributing to LeRobot Instruction Synthesis! This document provides guidelines and instructions for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lerobot-instruction-synthesis.git
   cd lerobot-instruction-synthesis
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev,test]"
   ```
5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🔧 Development Workflow

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure they follow our coding standards

3. **Run tests** to ensure nothing is broken:
   ```bash
   pytest tests/
   ```

4. **Run linters** to check code quality:
   ```bash
   black lesynthesis/
   isort lesynthesis/
   ruff lesynthesis/
   mypy lesynthesis/
   ```

5. **Commit your changes** with a descriptive commit message:
   ```bash
   git add .
   git commit -m "feat: add support for new robot type"
   ```

6. **Push to your fork** and create a pull request

## 📝 Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add support for multi-robot trajectory analysis
fix: correct timing calculation in low-level instructions
docs: update API documentation for generate_instructions
```

## 🧪 Testing

- Write tests for any new functionality
- Ensure all tests pass before submitting a PR
- Aim for at least 80% code coverage
- Use pytest fixtures for common test setups

Example test:
```python
def test_instruction_generation():
    tool = LLMEnrichmentTool()
    instructions = tool.generate_instructions(
        dataset_repo_id="test/dataset",
        episode_index=0
    )
    assert "high_level" in instructions
    assert "mid_level" in instructions
    assert "low_level" in instructions
```

## 📚 Documentation

- Update the README.md if you change functionality
- Add docstrings to all functions and classes
- Include type hints for better code clarity
- Update API documentation for public interfaces

## 🐛 Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Complete error traceback
- Minimal code example to reproduce the issue
- Dataset information (if applicable)

## 💡 Suggesting Enhancements

We welcome suggestions! Please:
- Check if the enhancement has already been suggested
- Provide a clear use case
- Explain the expected behavior
- Consider implementing it yourself!

## 🔍 Code Review Process

1. All submissions require review before merging
2. Reviewers will check for:
   - Code quality and style
   - Test coverage
   - Documentation updates
   - Performance implications
3. Address reviewer feedback promptly
4. Once approved, your PR will be merged!

## 🏗️ Project Structure

```
lerobot-instruction-synthesis/
├── lesynthesis/          # Main package code
│   ├── __init__.py
│   ├── enrich_with_llm.py      # Core functionality
│   └── enrich_with_llm_server.py # REST API server
├── tests/                # Test files
├── docs/                 # Documentation
├── examples/             # Example scripts
└── scripts/              # Utility scripts
```

## 📦 Adding Dependencies

If you need to add a new dependency:
1. Add it to the appropriate section in `pyproject.toml`
2. Document why it's needed
3. Ensure it's compatible with existing dependencies
4. Update installation instructions if needed

## 🎯 Areas for Contribution

- **New LLM Providers**: Add support for OpenAI, Anthropic, etc.
- **Performance Optimization**: Improve trajectory analysis speed
- **New Features**: Multi-language support, custom prompts
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Increase test coverage, add integration tests
- **UI/UX**: Improve web interface, add visualizations

## 📞 Getting Help

- Open an issue for bugs or questions
- Join our discussions on GitHub
- Check existing issues and PRs before starting work

Thank you for contributing to LeRobot Instruction Synthesis! 🤖✨ 