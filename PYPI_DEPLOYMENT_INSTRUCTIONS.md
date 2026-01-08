# PyPI Deployment Instructions

This document provides instructions for deploying the `llm-med` package to PyPI.

## Prerequisites

Before deploying to PyPI, you need to set up API tokens as GitHub Secrets.

### Required GitHub Secrets

You need to configure the following secrets in your GitHub repository:

1. **PYPI_API_TOKEN** (Required for production PyPI releases)
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token with the scope for this project
   - Copy the token (starts with `pypi-`)
   - Add it as a secret named `PYPI_API_TOKEN` in your GitHub repository

2. **TEST_PYPI_API_TOKEN** (Optional, for testing)
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new API token
   - Copy the token (starts with `pypi-`)
   - Add it as a secret named `TEST_PYPI_API_TOKEN` in your GitHub repository

### How to Add GitHub Secrets

1. Go to your repository on GitHub
2. Click on "Settings" tab
3. In the left sidebar, click "Secrets and variables" > "Actions"
4. Click "New repository secret"
5. Add each secret with the exact name as shown above

## Deployment Methods

### Method 1: Release via GitHub (Recommended)

1. Ensure all your changes are committed and pushed to the `main` branch
2. Create a new release on GitHub:
   - Go to your repository's "Releases" page
   - Click "Create a new release"
   - Create a new tag (e.g., `v0.1.0`)
   - Add release notes
   - Click "Publish release"
3. The GitHub Actions workflow will automatically:
   - Build the package
   - Upload to PyPI using your `PYPI_API_TOKEN`

### Method 2: Manual Workflow Trigger (For Testing)

1. Go to the "Actions" tab in your repository
2. Select the "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select the branch (usually `main`)
5. Click "Run workflow" button
6. This will:
   - Build the package
   - Upload to Test PyPI using your `TEST_PYPI_API_TOKEN`
   - Only runs if you have the TEST_PYPI_API_TOKEN configured

## Package Build Status

The package has been configured and tested to build successfully with:
- All Python code formatted with Black
- All critical flake8 errors resolved
- Proper package structure with all submodules included
- Valid package metadata

## Verification

After deployment, you can verify the package is available on PyPI:

```bash
# Check PyPI
pip search llm-med  # (Note: PyPI search may be disabled)
# or visit https://pypi.org/project/llm-med/

# Install the package
pip install llm-med

# Test the installation
python -c "import model; import inference; print('Package imported successfully!')"
```

## Troubleshooting

### "403 Forbidden" during upload
- Verify your API token is correct
- Ensure the token has the right scope (project or all projects)
- Check that the package name is not already taken by another user

### Package already exists
- Increment the version number in `pyproject.toml`
- You cannot overwrite a version that's already on PyPI
- Each release must have a unique version number

### Workflow fails due to missing secrets
- Verify the secrets are added with the exact names shown above
- Secrets are case-sensitive

## Version Management

Before each new release:

1. Update the version number in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment as needed
   ```

2. Update `CHANGELOG.md` with the changes in this version

3. Commit the changes:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 0.1.1"
   git push
   ```

4. Create a new release on GitHub with the matching tag (e.g., `v0.1.1`)

## Notes

- The GitHub Actions workflow automatically runs tests before deployment
- Code formatting and linting checks must pass before the package is built
- All dependencies are specified in `pyproject.toml`
- The package includes all necessary modules and data files
