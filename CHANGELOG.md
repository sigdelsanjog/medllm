# Changelog

All notable changes to the GptMed package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.3] - 2026-01-28

### Changed

- CLI: Improved project name validation in `startproject` to match Django's convention. Project names with dashes (-) are now disallowed; only valid Python identifiers are accepted.

---

## [0.4.0] - 2026-01-17

### Added

#### New Observability Module (`gptmed.observability`)

A comprehensive training observability system implementing the **Observer Pattern** for decoupled monitoring and XAI (Explainable AI) foundations.

**Base Classes & Data Structures:**

- `TrainingObserver` - Abstract base class for all training observers
- `ObserverManager` - Composite pattern manager for multiple observers
- `StepMetrics` - Data class for training step metrics (loss, grad_norm, lr, perplexity)
- `ValidationMetrics` - Data class for validation metrics
- `GradientMetrics` - Data class for gradient flow analysis (future XAI)
- `TrainingEvent` - Enum for training lifecycle events

**MetricsTracker:**

- Comprehensive metrics tracking with loss curve history
- Moving averages for smoothed visualization
- Perplexity tracking (train & validation)
- Learning rate schedule tracking
- Gradient norm monitoring
- Automatic issue detection:
  - NaN loss detection
  - Overfitting detection (val loss increasing)
  - Stalled training detection
  - Gradient explosion warnings
- Export capabilities:
  - `export_to_csv()` - Spreadsheet-ready format
  - `export_to_json()` - Full metrics export
  - `plot_loss_curves()` - 4-panel visualization (requires matplotlib)

**Training Callbacks:**

- `ConsoleCallback` - Pretty console output with progress and warnings
- `JSONLoggerCallback` - JSONL format logging for analysis
- `EarlyStoppingCallback` - Stop training on validation plateau
- `LRSchedulerCallback` - Learning rate monitoring and plateau detection

#### Trainer Integration

- Added `observers` parameter to `Trainer.__init__()`
- Added `add_observer()` method for dynamic observer registration
- Trainer now emits events at all lifecycle points:
  - `on_train_start` - Training begins with config
  - `on_epoch_start` / `on_epoch_end` - Epoch boundaries
  - `on_step` - Every training step with metrics
  - `on_validation` - After validation runs
  - `on_checkpoint` - When checkpoints are saved
  - `on_train_end` - Training completion with final metrics
- Early stopping support via observer `should_stop` flag

#### TrainingService Integration

- `execute_training()` now accepts optional `observers` parameter
- Default observers created automatically if none provided:
  - `MetricsTracker` enabled by default
- Automatic report generation on training completion:
  - CSV export
  - JSON export
  - Loss curve plots (if matplotlib installed)
  - Training health check printed to console
- Graceful handling of interrupted training (Ctrl+C):
  - Checkpoint saved
  - Observability reports still generated

#### Package Exports

New exports available at package level:

```python
from gptmed import (
    TrainingObserver,
    ObserverManager,
    MetricsTracker,
    ConsoleCallback,
    JSONLoggerCallback,
    EarlyStoppingCallback,
)
```

### Changed

- `Trainer.train_step()` now accepts `step` and `lr` parameters for observer metrics
- `Trainer.evaluate()` now notifies observers with `ValidationMetrics`
- `Trainer.train()` refactored with `_finish_training()` helper for clean exit handling
- `TrainingService.execute_training()` signature extended with `observers` parameter
- Checkpoint naming changed from `best_model.pt` to `final_model.pt`

### Files Added

```
gptmed/gptmed/observability/
├── __init__.py           # Module exports
├── base.py               # TrainingObserver, ObserverManager, data classes
├── metrics_tracker.py    # MetricsTracker implementation
└── callbacks.py          # Console, JSON, EarlyStopping, LRScheduler callbacks
```

### Design Patterns Used

| Pattern                  | Implementation                                                            |
| ------------------------ | ------------------------------------------------------------------------- |
| **Observer**             | `TrainingObserver` → Trainer emits events, observers react                |
| **Composite**            | `ObserverManager` → manages multiple observers uniformly                  |
| **Strategy**             | Swap logging backends (Console/JSON/TensorBoard) without changing Trainer |
| **Template Method**      | Abstract methods in `TrainingObserver`, concrete in callbacks             |
| **Dependency Inversion** | Trainer depends on `TrainingObserver` abstraction                         |

### Usage Example

```python
from gptmed.training.trainer import Trainer
from gptmed.observability import MetricsTracker, ConsoleCallback, EarlyStoppingCallback

# Create observers
tracker = MetricsTracker(log_dir="logs/experiment1", moving_avg_window=100)
console = ConsoleCallback(log_interval=100)
early_stop = EarlyStoppingCallback(patience=5)

# Create trainer with observers
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    config=train_config,
    device="cpu",
    observers=[tracker, console, early_stop]
)

# Train - observers notified automatically
trainer.train()

# After training - access metrics
tracker.plot_loss_curves()
tracker.export_to_csv("metrics.csv")
issues = tracker.detect_issues()
```

### Output Files Generated

After training completes, the following files are created in `log_dir/`:

| File                                | Description                     |
| ----------------------------------- | ------------------------------- |
| `gptmed_training_metrics.jsonl`     | Step-by-step metrics (JSONL)    |
| `gptmed_training_config.json`       | Training configuration snapshot |
| `gptmed_training_summary.json`      | Final training summary          |
| `gptmed_training_metrics.csv`       | All metrics in CSV format       |
| `gptmed_training_full_metrics.json` | Complete metrics export         |
| `gptmed_training_loss_curves.png`   | Loss curve plots (4 panels)     |

---

## [0.3.3] - Previous Release

- Initial stable release with training and inference capabilities
- High-level API (`create_config`, `train_from_config`, `generate`)
- DeviceManager for flexible device selection
- TrainingService for orchestrated training

---

## Future Roadmap (XAI Features)

The observability module is designed to support future XAI capabilities:

- [ ] **Attention Visualization** - See what tokens the model focuses on
- [ ] **Saliency Maps** - Input attribution using Integrated Gradients
- [ ] **Logit Lens** - Layer-by-layer prediction analysis
- [ ] **Embedding Space Analysis** - t-SNE visualization of learned concepts
- [ ] **Gradient Flow Analysis** - Detect vanishing/exploding gradients per layer
- [ ] **TensorBoard Integration** - Real-time training dashboards
- [ ] **Weights & Biases Integration** - Experiment tracking

See `XAI.md` in the project root for detailed implementation plans.
