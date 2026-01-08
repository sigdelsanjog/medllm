"""
Basic tests for the llm-med package.
"""

import pytest


def test_package_structure():
    """Test that the package structure is correct."""
    import sys
    from pathlib import Path

    # Test that main modules exist
    package_dir = Path(__file__).parent.parent
    assert (package_dir / "model" / "__init__.py").exists()
    assert (package_dir / "inference" / "__init__.py").exists()
    assert (package_dir / "training" / "__init__.py").exists()
    assert (package_dir / "utils" / "__init__.py").exists()
    assert (package_dir / "configs" / "__init__.py").exists()
    assert (package_dir / "data" / "__init__.py").exists()
    assert (package_dir / "tokenizer" / "__init__.py").exists()


def test_subpackages_exist():
    """Test that subpackages exist."""
    from pathlib import Path

    package_dir = Path(__file__).parent.parent
    assert (package_dir / "model" / "architecture" / "__init__.py").exists()
    assert (package_dir / "model" / "configs" / "__init__.py").exists()
    assert (package_dir / "data" / "parsers" / "__init__.py").exists()


def test_generation_config_import():
    """Test that generation configuration can be imported without dependencies."""
    # This should work as it doesn't require torch
    from inference.generation_config import GenerationConfig

    assert GenerationConfig is not None


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and has the right package name."""
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.exists()

    content = pyproject.read_text()
    assert 'name = "llm-med"' in content
    assert "version" in content

