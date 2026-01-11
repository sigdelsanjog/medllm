# Changed Files for GPTMed v0.3.3 Deployment

## Core Package Files (gptmed/) - NEEDS DEPLOYMENT

These files have been modified and need to be deployed to PyPI:

### 1. Version Files

- **`gptmed/__init__.py`**
  - Updated version: `0.3.2` → `0.3.3` (needs update)
- **`pyproject.toml`**
  - Updated version: `0.3.2` → `0.3.3` (needs update)

### 2. Configuration Files

- **`gptmed/configs/config_loader.py`**
  - ✅ Fixed parameter naming: `save_every` → `save_interval`
  - ✅ Fixed parameter naming: `eval_every` → `eval_interval`
  - ✅ Fixed parameter naming: `log_every` → `log_interval`
  - ✅ Added backward compatibility with `.get()` for both old and new names
  - Lines changed: ~113-120, ~163-169

### 3. API Files

- **`gptmed/api.py`**
  - ✅ Fixed TrainingConfig parameter names (save_interval, eval_interval, log_interval)
  - ✅ Fixed TextGenerator initialization (load tokenizer object)
  - ✅ Fixed generate() to use GenerationConfig object
  - Lines changed: ~190, ~337-358

## Summary of Bug Fixes

### Issue 1: Parameter Naming Mismatch

**Problem**: YAML config template used `save_interval`, `eval_interval`, `log_interval` but code tried to read `save_every`, `eval_every`, `log_every`

**Fix**:

- config_loader.py: Use `.get()` to support both naming conventions
- api.py: Use correct parameter names when creating TrainingConfig

### Issue 2: TextGenerator Initialization

**Problem**: api.py passed `tokenizer_path` string, but TextGenerator expects tokenizer object

**Fix**:

- Load SentencePiece tokenizer before creating TextGenerator
- Pass tokenizer object instead of path

### Issue 3: Generation Config

**Problem**: api.py called generator.generate() with individual parameters, but it expects GenerationConfig object

**Fix**:

- Create GenerationConfig object with parameters
- Pass config object to generator.generate()

## Files to Deploy

### Required for Deployment:

```
gptmed/
├── gptmed/
│   ├── __init__.py          # Update version to 0.3.3
│   ├── api.py               # Fixed generation and training
│   └── configs/
│       └── config_loader.py # Fixed parameter naming
└── pyproject.toml           # Update version to 0.3.3
```

### Deployment Steps:

1. **Update version numbers**:

   ```bash
   # In pyproject.toml: version = "0.3.3"
   # In gptmed/__init__.py: __version__ = "0.3.3"
   ```

2. **Commit changes**:

   ```bash
   cd gptmed
   git add -A
   git commit -m "Fix v0.3.3: Fix parameter naming, TextGenerator init, and GenerationConfig usage"
   ```

3. **Create and push tag**:

   ```bash
   git tag -a v0.3.3 -m "Bug fixes: parameter naming, tokenizer loading, generation config"
   git push origin main
   git push origin v0.3.3
   ```

4. **GitHub Actions will automatically**:
   - Build the package
   - Upload to PyPI
   - Create GitHub release

## Testing Files (gptmed-api/) - NOT FOR DEPLOYMENT

These are test/demo files in gptmed-api folder:

- `test_training_api.py` - Test training functionality
- `test_generation_api.py` - Test generation functionality
- `test_complete_workflow.py` - End-to-end workflow test
- `example_config.yaml` - Pre-configured example
- `DATA_GUIDE.md` - Data files documentation
- `EXAMPLES.md` - Usage examples
- `README.md` - Updated documentation

## Verification After Deployment

After v0.3.3 is deployed to PyPI:

```bash
# Update local installation
pip install --upgrade gptmed

# Verify version
python -c "import gptmed; print(gptmed.__version__)"
# Should print: 0.3.3

# Test training
cd gptmed-api
python test_training_api.py

# Test generation
python test_generation_api.py

# Test complete workflow
python test_complete_workflow.py
```

## Backward Compatibility

The config_loader.py now supports both naming conventions:

- Old: `save_every`, `eval_every`, `log_every`
- New: `save_interval`, `eval_interval`, `log_interval`

This ensures existing config files will still work.
