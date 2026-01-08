# PyPI Deployment Guide for llm-med

This guide will walk you through deploying your `medllm` package to PyPI so users can install it with `pip install llm-med`.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Prepare Your Package](#prepare-your-package)
3. [Set Up PyPI Accounts](#set-up-pypi-accounts)
4. [Local Testing](#local-testing)
5. [GitHub Repository Setup](#github-repository-setup)
6. [Configure GitHub Secrets](#configure-github-secrets)
7. [Publishing to PyPI](#publishing-to-pypi)
8. [Using Your Package](#using-your-package)

---

## Prerequisites

### 1. Install Build Tools

```bash
pip install --upgrade build twine wheel setuptools
```

### 2. Required Accounts

- **PyPI Account**: https://pypi.org/account/register/
- **Test PyPI Account** (optional but recommended): https://test.pypi.org/account/register/
- **GitHub Account**: For repository hosting and GitHub Actions

---

## Prepare Your Package

### 1. Update Package Information

Edit [`setup.py`](setup.py) and [`pyproject.toml`](pyproject.toml):

```python
# In setup.py and pyproject.toml, update:
name="llm-med"  # Your package name (must be unique on PyPI)
version="0.1.0"  # Increment for each release
author="Your Name"
author_email="your.email@example.com"
url="https://github.com/yourusername/medllm"
```

### 2. Create Proper README

Rename `README_PACKAGE.md` to `README.md`:

```bash
mv README_PACKAGE.md README.md
```

Update the README with:

- Your actual GitHub username/repo URL
- Real examples and documentation
- Performance metrics from your training

### 3. Package Structure

Ensure your package has proper `__init__.py` files in all module directories:

```bash
# Create __init__.py files if missing
touch model/__init__.py
touch model/architecture/__init__.py
touch model/configs/__init__.py
touch inference/__init__.py
touch training/__init__.py
touch utils/__init__.py
touch configs/__init__.py
```

### 4. Update Version Number

Before each release, update version in:

- `setup.py` → `version="0.1.0"`
- `pyproject.toml` → `version = "0.1.0"`
- `CHANGELOG.md` → Add new version section

---

## Set Up PyPI Accounts

### 1. Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Verify your email
3. Enable 2FA (recommended)

### 2. Create API Tokens

**For PyPI:**

1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name: `github-actions-llm-med`
5. Scope: "Entire account" (or specific to `llm-med` after first upload)
6. **COPY THE TOKEN** (you won't see it again!)

**For Test PyPI (optional):**

1. Go to https://test.pypi.org/manage/account/
2. Repeat the same process
3. Name: `github-actions-llm-med-test`

---

## Local Testing

### 1. Build the Package Locally

```bash
cd /path/to/medllm

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build package
python -m build
```

This creates:

- `dist/llm_med-0.1.0-py3-none-any.whl` (wheel distribution)
- `dist/llm-med-0.1.0.tar.gz` (source distribution)

### 2. Check Package Quality

```bash
# Validate package metadata
twine check dist/*

# Should output: PASSED
```

### 3. Test Installation Locally

```bash
# Install in development mode
pip install -e .

# Or install from built wheel
pip install dist/llm_med-0.1.0-py3-none-any.whl
```

### 4. Test Import

```bash
python -c "from model.architecture import GPTTransformer; print('Success!')"
python -c "from inference.generator import MedicalQAGenerator; print('Success!')"
```

### 5. Upload to Test PyPI (Optional but Recommended)

```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Enter your Test PyPI credentials or API token:
# Username: __token__
# Password: <your-test-pypi-token>

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ llm-med
```

---

## GitHub Repository Setup

### 1. Create New Repository

```bash
# On GitHub, create a new repository named "medllm"
# Then locally:

cd /path/to/medllm

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: llm-med package v0.1.0"

# Add remote
git remote add origin https://github.com/yourusername/medllm.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2. Create .gitignore

Create [`.gitignore`](.gitignore):

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# PyCharm
.idea/

# VSCode
.vscode/

# Jupyter
.ipynb_checkpoints

# Training artifacts
logs/
model/checkpoints/
dataset/
data/processed/
data/tokenized/

# Environment variables
.env

# OS
.DS_Store
Thumbs.db
EOF
```

---

## Configure GitHub Secrets

### 1. Add PyPI Token to GitHub

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add two secrets:

**Secret 1:**

- Name: `PYPI_API_TOKEN`
- Value: `<your-pypi-api-token>`

**Secret 2 (optional):**

- Name: `TEST_PYPI_API_TOKEN`
- Value: `<your-test-pypi-api-token>`

### 2. Verify GitHub Actions is Enabled

1. Go to **Settings** → **Actions** → **General**
2. Ensure "Allow all actions and reusable workflows" is selected

---

## Publishing to PyPI

### Method 1: Automatic Publishing via GitHub Release (Recommended)

1. **Commit and push all changes:**

```bash
git add .
git commit -m "Prepare v0.1.0 release"
git push origin main
```

2. **Create a Git Tag:**

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

3. **Create a GitHub Release:**

- Go to your repository on GitHub
- Click **Releases** → **Create a new release**
- Tag: `v0.1.0`
- Release title: `v0.1.0 - Initial Release`
- Description: Copy from CHANGELOG.md
- Click **Publish release**

4. **GitHub Actions will automatically:**

   - Build your package
   - Run tests
   - Upload to PyPI

5. **Monitor the workflow:**
   - Go to **Actions** tab
   - Watch "Publish to PyPI" workflow
   - Check for any errors

### Method 2: Manual Publishing via Terminal

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*

# Enter credentials:
# Username: __token__
# Password: <your-pypi-api-token>
```

### Method 3: Manual GitHub Actions Trigger

1. Go to **Actions** tab
2. Select "Publish to PyPI" workflow
3. Click **Run workflow**
4. Select branch: `main`
5. Click **Run workflow**

This publishes to **Test PyPI** (good for testing without affecting production PyPI)

---

## Using Your Package

### For Other Developers

Once published to PyPI, anyone can install your package:

```bash
pip install llm-med
```

### In Your Current Project

After publishing, you can use it in your `code-llm` project:

**In [`backend/main.py`](../backend/main.py):**

```python
# Remove local imports, use the package
from llm_med.model.architecture import GPTTransformer
from llm_med.inference.generator import MedicalQAGenerator
from llm_med.model.configs.model_config import get_small_config

# Your backend code...
```

**Install in your backend:**

```bash
cd /path/to/code-llm/backend
pip install llm-med
```

---

## Version Management

### Semantic Versioning

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): New functionality, backwards-compatible
- **PATCH** version (0.0.1): Bug fixes, backwards-compatible

### Release Checklist

Before each new release:

- [ ] Update version in `setup.py`
- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests: `pytest tests/`
- [ ] Build locally: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Test locally: `pip install dist/*.whl`
- [ ] Commit changes
- [ ] Create git tag: `git tag v0.x.x`
- [ ] Push tag: `git push origin v0.x.x`
- [ ] Create GitHub release

---

## Troubleshooting

### Package Name Already Exists

```
Error: Package name already exists on PyPI
```

**Solution:** Choose a different name in `setup.py` and `pyproject.toml`:

- `llm-med-custom`
- `medllm-qa`
- `medical-llm-<yourname>`

### Authentication Errors

```
Error: Invalid credentials
```

**Solution:**

- Use `__token__` as username
- Ensure API token is correct
- Check token hasn't expired

### GitHub Actions Fails

**Check:**

1. Secrets are properly set in repository settings
2. `publish-to-pypi.yml` workflow file is in `.github/workflows/`
3. Check Actions logs for specific errors

### Import Errors After Installation

```
ImportError: No module named 'model'
```

**Solution:**

- Ensure all directories have `__init__.py`
- Check `MANIFEST.in` includes necessary files
- Verify `setup.py` `packages` parameter

---

## Best Practices

### 1. Always Test Before Production

```bash
# Always test on Test PyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ llm-med
```

### 2. Use Versioning Tags

```bash
# Tag every release
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

### 3. Keep CHANGELOG Updated

Document all changes in `CHANGELOG.md` for transparency.

### 4. Security

- **Never** commit API tokens to git
- Use GitHub Secrets for all credentials
- Enable 2FA on PyPI

### 5. Documentation

- Keep README.md updated with examples
- Document all breaking changes
- Provide migration guides for major versions

---

## Next Steps

After successful deployment:

1. **Announce your package:**

   - Share on Reddit: r/Python, r/MachineLearning
   - Twitter/X with #Python #NLP #MedicalAI
   - Dev.to blog post

2. **Add badges to README:**

   ```markdown
   [![PyPI version](https://badge.fury.io/py/llm-med.svg)](https://pypi.org/project/llm-med/)
   [![Downloads](https://pepy.tech/badge/llm-med)](https://pepy.tech/project/llm-med)
   ```

3. **Monitor usage:**

   - Check PyPI download stats
   - Respond to GitHub issues
   - Accept community contributions

4. **Continuous improvement:**
   - Regular updates and bug fixes
   - Community feedback integration
   - Performance improvements

---

## Quick Reference Commands

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Install locally for testing
pip install -e .

# Create and push tag
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0

# Clean build artifacts
rm -rf build/ dist/ *.egg-info
```

---

## Support

If you encounter issues:

- Check [PyPI Documentation](https://packaging.python.org/)
- Review [GitHub Actions Docs](https://docs.github.com/en/actions)
- Open an issue on your repository

---

**Good luck with your package deployment! 🚀**
