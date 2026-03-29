---
name: model-training
description: Train and evaluate ML/NLP models with proper experiment tracking
---

# Model Training Workflow

Train and evaluate a model: $ARGUMENTS

## Steps

1. **Config**: Define hyperparameters in a config file or dataclass
2. **Data**: Load processed data from `corpus/processed/`, verify splits exist
3. **Model**: Initialize model architecture, log parameter count
4. **Train**: Run training loop with validation, log metrics each epoch
5. **Evaluate**: Run evaluation on test set, compute task-specific metrics
6. **Save**: Save model checkpoint and config to `models/`
7. **Report**: Print summary — metrics, training time, best epoch

## Guidelines

- Always set random seeds for reproducibility
- Use early stopping to avoid overfitting
- Log GPU/memory usage if applicable
- Save the full config alongside the model so experiments are reproducible
- Never commit model checkpoints to git (they belong in `.gitignore`)
