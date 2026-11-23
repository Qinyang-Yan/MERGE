# Reproducibility and release checks

## Artifact bundle

Treat the following files as one inseparable run bundle:

- `reports/<run>_split.json`
- `preprocessing/gene_index.csv`
- `preprocessing/gene_selection_<run>.json`
- `preprocessing/gene_mapping_<run>.json`
- `preprocessing/metal_feature_stats_<run>.json`
- `preprocessing/class_weights_<run>.json`
- `pseudobulk/<run>/metal_features_standardized.npz`
- `checkpoints/<run>/config.yaml`
- `checkpoints/<run>/best.pt`

The evaluation commands require the run-specific descriptor artifact and do not fall back to another run's files.

## Data isolation

The split is created from biological sample identifiers before HVG fitting. Validation and test perturbation cells are excluded from gene filtering/HVG selection, descriptor-scaler fitting, and class-weight estimation. Control cells form the shared reference used by the model protocol.

Checkpoint selection uses validation data only. Test data are evaluated after training with the frozen `best.pt`. External evaluation loads the same frozen checkpoint and does not optimize model parameters.

## Randomness

Set and report the run seed. Control pairing uses seeded NumPy/DataLoader worker generators. External control samples are generated with an explicit local generator and stored predictions can be audited through vote counts and group sizes.

For strict replication, record:

- Python, PyTorch, CUDA, and operating-system versions;
- all dependencies from `python -m pip freeze`;
- the GeneCompass source commit and asset checksums;
- the resolved YAML configuration and command-line overrides;
- input data checksums and descriptor table versions.

Example checksum command on macOS:

```bash
shasum -a 256 outputs/checkpoints/merge/best.pt
```

## Recommended release verification

1. Build a clean environment from `pyproject.toml`.
2. Run `python -m pytest -q`.
3. Run every CLI with `--help`.
4. Execute the full pipeline from raw inputs in the clean environment.
5. Repeat deterministic evaluation and compare metrics and predictions.
6. Verify that all reported figures and tables are generated from saved run artifacts rather than manually entered values.
7. Archive the complete artifact bundle and environment record alongside any released checkpoint.

## Checkpoint and artifact compatibility

The release protocol fits HVGs after sample splitting, uses seeded control pairing, advances the scheduler at optimizer updates, and computes Top-K metrics from the three-class labels.

Generate every checkpoint and its reported metrics through this protocol with one frozen artifact bundle. Strict checkpoint loading verifies tensor names and shapes. Evaluation requires the checkpoint's ordered gene index and matching processed control/reference artifacts.

## Automated test coverage

The unit suite uses small in-memory arrays and a lightweight cell encoder for model-assembly tests. Full GeneCompass integration and end-to-end experimental analysis are separate verification steps listed above.
