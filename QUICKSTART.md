# Quick Start Guide for MedLLM Package Repository

This directory contains everything needed to deploy `llm-med` to PyPI.

## Directory Structure

```
medllm-package/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # CI/CD testing
│       └── publish-to-pypi.yml       # PyPI deployment
├── model/                             # Core model architecture
├── inference/                         # Text generation
├── training/                          # Training scripts
├── utils/                            # Utilities
├── configs/                          # Configuration
├── data/                             # Data processing
├── tokenizer/                        # Tokenizer
├── setup.py                          # Package setup (legacy)
├── pyproject.toml                    # Package setup (modern)
├── MANIFEST.in                       # Package file inclusion rules
├── requirements.txt                  # Dependencies
├── README.md                         # Package documentation
├── LICENSE                           # MIT License
├── CHANGELOG.md                      # Version history
├── DEPLOYMENT_GUIDE.md               # Detailed deployment instructions
└── .gitignore                        # Git ignore rules
```

## Steps to Deploy

### 1. Initialize Git Repository

```bash
cd medllm-package
git init
git add .
git commit -m "Initial commit: llm-med package v0.1.0"
```

### 2. Connect to GitHub

```bash
# Add your remote repository
git remote add origin https://github.com/yourusername/medllm.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Update Package Information

Edit these files with your information:

**setup.py:**

- Line 19: `author="Your Name"`
- Line 20: `author_email="your.email@example.com"`
- Line 23: `url="https://github.com/yourusername/medllm"`

**pyproject.toml:**

- Line 10: `{name = "Your Name", email = "your.email@example.com"}`
- Line 50: `Homepage = "https://github.com/yourusername/medllm"`

**README.md:**

- Update all GitHub URLs with your username

### 4. Set Up PyPI

1. Create account at https://pypi.org/account/register/
2. Generate API token at https://pypi.org/manage/account/
3. Add token to GitHub Secrets as `PYPI_API_TOKEN`

### 5. Test Locally

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Test install
pip install dist/*.whl
```

### 6. Deploy to PyPI

**Option A: Via GitHub Release (Recommended)**

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
# Then create a release on GitHub
```

**Option B: Manual**

```bash
twine upload dist/*
```

## After Deployment

Install in your main project:

```bash
cd /path/to/code-llm/backend
pip install llm-med
```

Use in your code:

```python
from llm_med.model.architecture import GPTTransformer
from llm_med.inference.generator import MedicalQAGenerator
```

## Documentation

- Full deployment guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Package README: [README.md](README.md)
- Version history: [CHANGELOG.md](CHANGELOG.md)

## Important Files

- **DO NOT commit**: Large model files (.pt, .pth), datasets, logs
- **DO commit**: Source code, configs, documentation, workflows
- **Update before release**: Version numbers, CHANGELOG.md

## Troubleshooting

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed troubleshooting.
