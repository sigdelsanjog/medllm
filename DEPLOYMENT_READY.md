# PyPI Deployment - Quick Start Guide

## ✅ What Has Been Fixed

The following issues have been resolved to enable successful PyPI deployment:

1. **Code Formatting**: All Python files formatted with Black
2. **Code Quality**: Critical flake8 linting errors fixed
3. **Package Structure**: All subpackages (model.architecture, data.parsers, etc.) now included
4. **Metadata Configuration**: Package metadata properly configured in pyproject.toml
5. **Build System**: Package successfully builds both wheel and source distribution
6. **Tests**: Basic test suite added to verify package integrity
7. **CI Workflow**: GitHub Actions workflow configured for automated deployment

## 🚀 Deployment Status

**Current Status**: ✅ Ready for Deployment

The package is now ready to be deployed to PyPI. All that's needed is to configure the PyPI API token.

## 📋 Required Action

**YOU NEED TO ADD THE FOLLOWING SECRET TO YOUR GITHUB REPOSITORY:**

### Step 1: Get Your PyPI API Token

1. Go to https://pypi.org and log in (create an account if you don't have one)
2. Click on your username in the top right, then "Account settings"
3. Scroll down to the "API tokens" section
4. Click "Add API token"
5. Give it a name (e.g., "medllm-deployment")
6. Set scope to "Entire account" (or create the project first and scope to that project)
7. Click "Add token"
8. **IMPORTANT**: Copy the token that starts with `pypi-` - you won't be able to see it again!

### Step 2: Add Secret to GitHub

1. Go to your repository: https://github.com/sigdelsanjog/medllm
2. Click "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `PYPI_API_TOKEN`
6. Value: Paste the token you copied from PyPI
7. Click "Add secret"

## 🎯 How to Deploy

Once you've added the secret, you have two deployment options:

### Option 1: Create a GitHub Release (Recommended)

1. Go to your repository's main page
2. Click "Releases" (right sidebar)
3. Click "Create a new release" or "Draft a new release"
4. Click "Choose a tag"
5. Type a version tag (e.g., `v0.1.0`) and click "Create new tag"
6. Add a release title (e.g., "Initial Release v0.1.0")
7. Add release notes describing what's in this version
8. Click "Publish release"

The GitHub Actions workflow will automatically:
- Build the package
- Run tests
- Upload to PyPI

### Option 2: Manual Workflow Trigger (For Testing with Test PyPI)

1. Go to the "Actions" tab in your repository
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select branch (usually `main`)
5. Click "Run workflow"

This will upload to Test PyPI (https://test.pypi.org) for testing purposes.

## 📦 After Deployment

Once deployed, users can install your package with:

```bash
pip install llm-med
```

You can verify deployment at: https://pypi.org/project/llm-med/

## 🔄 Future Releases

For future releases:

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment appropriately
   ```

2. Update `CHANGELOG.md` with changes

3. Commit and push changes

4. Create a new GitHub release with the new version tag

The workflow will automatically handle the rest!

## ❓ Troubleshooting

### "Package name already taken"
- Choose a different name in `pyproject.toml`
- Check availability at https://pypi.org/project/YOUR-NAME/

### "Invalid credentials"
- Verify the API token is correct
- Check the secret name is exactly `PYPI_API_TOKEN`
- Ensure the token hasn't been revoked

### "Version already exists"
- Each release must have a unique version number
- Increment the version in `pyproject.toml` before releasing

## 📚 Additional Documentation

- Full deployment guide: See `PYPI_DEPLOYMENT_INSTRUCTIONS.md`
- Package documentation: See `README.md`
- Changelog: See `CHANGELOG.md`
