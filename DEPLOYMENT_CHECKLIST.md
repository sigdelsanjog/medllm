# MedLLM Package Deployment Checklist

Use this checklist to track your deployment progress.

## ✅ Pre-Deployment Setup

- [ ] **Update package information in files:**

  - [ ] `setup.py` - lines 19-23 (author, email, URL)
  - [ ] `pyproject.toml` - lines 10, 50 (author, URLs)
  - [ ] `README.md` - all GitHub URLs with your username

- [ ] **Review and customize:**

  - [ ] README.md - add real examples and documentation
  - [ ] CHANGELOG.md - update with accurate dates and changes
  - [ ] LICENSE - update copyright year and name

- [ ] **Verify package name availability:**
  - [ ] Check https://pypi.org/project/llm-med/ (should be 404)
  - [ ] If taken, choose alternative name in setup.py and pyproject.toml

## ✅ GitHub Repository Setup

- [ ] **Create GitHub repository:**

  - [ ] Repository name: `medllm`
  - [ ] Description: "Medical QA language model package"
  - [ ] Public or Private (Public recommended for PyPI)

- [ ] **Initialize and push:**

  ```bash
  cd medllm-package
  git init
  git add .
  git commit -m "Initial commit: llm-med package v0.1.0"
  git remote add origin https://github.com/YOURUSERNAME/medllm.git
  git branch -M main
  git push -u origin main
  ```

- [ ] **Verify GitHub:**
  - [ ] Files are visible on GitHub
  - [ ] .github/workflows/ directory is present
  - [ ] README.md displays correctly

## ✅ PyPI Account Setup

- [ ] **Create PyPI account:**

  - [ ] Sign up at https://pypi.org/account/register/
  - [ ] Verify email address
  - [ ] Enable 2FA (recommended)

- [ ] **Generate API tokens:**

  - [ ] PyPI API token (https://pypi.org/manage/account/)
  - [ ] Name: `github-actions-llm-med`
  - [ ] Scope: "Entire account" initially
  - [ ] Copy token (starts with `pypi-`)

- [ ] **Optional: Test PyPI account:**
  - [ ] Sign up at https://test.pypi.org/account/register/
  - [ ] Generate Test PyPI API token
  - [ ] Name: `github-actions-llm-med-test`

## ✅ GitHub Secrets Configuration

- [ ] **Add secrets to GitHub repository:**

  - [ ] Go to: Settings → Secrets and variables → Actions
  - [ ] New repository secret: `PYPI_API_TOKEN`
  - [ ] Value: Your PyPI API token
  - [ ] Optional: New repository secret: `TEST_PYPI_API_TOKEN`

- [ ] **Verify GitHub Actions is enabled:**
  - [ ] Settings → Actions → General
  - [ ] "Allow all actions and reusable workflows" is selected

## ✅ Local Testing

- [ ] **Install build tools:**

  ```bash
  pip install --upgrade build twine wheel setuptools
  ```

- [ ] **Build package locally:**

  ```bash
  cd medllm-package
  rm -rf build/ dist/ *.egg-info
  python -m build
  ```

- [ ] **Verify build output:**

  - [ ] `dist/llm_med-0.1.0-py3-none-any.whl` exists
  - [ ] `dist/llm-med-0.1.0.tar.gz` exists

- [ ] **Check package quality:**

  ```bash
  twine check dist/*
  ```

  - [ ] Output shows "PASSED"

- [ ] **Test local installation:**

  ```bash
  pip install -e .
  # Or: pip install dist/llm_med-0.1.0-py3-none-any.whl
  ```

- [ ] **Test imports:**

  ```bash
  python -c "from model.architecture import GPTTransformer; print('✓ Model import works')"
  python -c "from inference.generator import MedicalQAGenerator; print('✓ Inference import works')"
  python -c "from training.train import main; print('✓ Training import works')"
  ```

- [ ] **Uninstall test:**
  ```bash
  pip uninstall llm-med -y
  ```

## ✅ Test PyPI Deployment (Optional but Recommended)

- [ ] **Upload to Test PyPI:**

  ```bash
  twine upload --repository testpypi dist/*
  ```

- [ ] **Install from Test PyPI:**

  ```bash
  pip install --index-url https://test.pypi.org/simple/ llm-med
  ```

- [ ] **Verify Test PyPI installation:**

  - [ ] Package installs without errors
  - [ ] Imports work correctly
  - [ ] Basic functionality works

- [ ] **Clean up:**
  ```bash
  pip uninstall llm-med -y
  ```

## ✅ Production PyPI Deployment

### Method 1: Via GitHub Release (Recommended)

- [ ] **Create and push tag:**

  ```bash
  git tag -a v0.1.0 -m "Release version 0.1.0"
  git push origin v0.1.0
  ```

- [ ] **Create GitHub Release:**

  - [ ] Go to: Repository → Releases → Create a new release
  - [ ] Choose tag: v0.1.0
  - [ ] Release title: "v0.1.0 - Initial Release"
  - [ ] Description: Copy from CHANGELOG.md
  - [ ] Click "Publish release"

- [ ] **Monitor GitHub Actions:**

  - [ ] Go to: Actions tab
  - [ ] Watch "Publish to PyPI" workflow
  - [ ] Verify it completes successfully (green checkmark)

- [ ] **Verify PyPI upload:**
  - [ ] Visit https://pypi.org/project/llm-med/
  - [ ] Package page displays correctly
  - [ ] Version 0.1.0 is shown

### Method 2: Manual Upload

- [ ] **Upload manually:**
  ```bash
  twine upload dist/*
  # Username: __token__
  # Password: <your-pypi-api-token>
  ```

## ✅ Post-Deployment Verification

- [ ] **Install from PyPI:**

  ```bash
  pip install llm-med
  ```

- [ ] **Verify installation:**

  ```bash
  pip show llm-med
  python -c "from llm_med.model.architecture import GPTTransformer; print('✓ Works!')"
  ```

- [ ] **Test in your main project:**
  - [ ] Navigate to: `/home/travelingnepal/Documents/proj/codellm/code-llm/backend`
  - [ ] Install: `pip install llm-med`
  - [ ] Update imports in backend code
  - [ ] Test backend functionality

## ✅ Documentation and Promotion

- [ ] **Update repository:**

  - [ ] Add PyPI badges to README.md
  - [ ] Update documentation with installation instructions
  - [ ] Add usage examples

- [ ] **Optional promotion:**
  - [ ] Share on Reddit (r/Python, r/MachineLearning)
  - [ ] Tweet about release
  - [ ] Write blog post on Dev.to

## ✅ Maintenance

- [ ] **Set up monitoring:**

  - [ ] Watch GitHub repository for issues
  - [ ] Monitor PyPI download stats
  - [ ] Set up notifications for CI/CD failures

- [ ] **Plan next release:**
  - [ ] Create issues for planned features
  - [ ] Update CHANGELOG.md for unreleased changes

---

## Quick Commands Reference

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Test PyPI upload
twine upload --repository testpypi dist/*

# Production PyPI upload
twine upload dist/*

# Create and push tag
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Clean build artifacts
rm -rf build/ dist/ *.egg-info
```

---

## Troubleshooting

If you encounter issues, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed troubleshooting steps.

---

**Last Updated:** 2026-01-08
