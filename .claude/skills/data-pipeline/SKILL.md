---
name: data-pipeline
description: Process corpus data — load, clean, tokenize, and split datasets
---

# Data Pipeline Workflow

Process corpus data for the mozhi-ai project: $ARGUMENTS

## Steps

1. **Inventory**: List files in `corpus/` and identify formats (txt, json, csv, parquet)
2. **Load**: Read raw data using appropriate loaders
3. **Clean**: Normalize text — handle encoding, whitespace, special characters
4. **Validate**: Check data quality — missing values, duplicates, malformed entries
5. **Transform**: Apply task-specific processing (tokenization, segmentation, labeling)
6. **Split**: Create train/val/test splits with stratification if applicable
7. **Save**: Write processed data to `corpus/processed/` in a reproducible format
8. **Document**: Log dataset statistics (size, vocab, class distribution)

## Guidelines

- Always preserve raw data — never modify files in `corpus/raw/`
- Use streaming/chunked processing for large files
- Log processing stats to stdout
- Write reproducible scripts (set random seeds for splits)
