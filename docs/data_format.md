# Data format and alignment

## Condition manifest

The training configuration accepts either unique filename discovery or an explicit `h5ad_files` mapping. An explicit mapping is recommended for reproducible analysis because it prevents accidental selection of a similarly named file.

```yaml
h5ad_files:
  Ctr: /path/to/Ctr.h5ad
  Mg: /path/to/Mg.h5ad
  Mg_alloy: /path/to/Mg_alloy.h5ad
  Mg_coat: /path/to/Mg_coat.h5ad
  Mo: /path/to/Mo.h5ad
  Ti: /path/to/Ti.h5ad
  Zn: /path/to/Zn.h5ad
```

Paths in YAML are absolute or resolved relative to the command's working directory. Environment-based defaults are available through `MERGE_DATA_ROOT`, `MERGE_OUTPUT_ROOT`, and `MERGE_ASSET_ROOT`.

## Expression matrices

The preferred representation is an AnnData H5AD with cells as observations and genes as variables. Counts must be non-negative raw counts in `X` or `layers['counts']`. If both are present, normalization starts from `layers['counts']`.

A reader is also provided for the project's R-exported HDF5 layout, where dense matrices are stored as genes by cells. This reader supports only that documented layout and requires explicit cell and gene indices.

## Required annotations

- `cell_type` (or `celltype`): reviewed biological annotation used for control matching and target grouping.
- A biological sample field: `sample_id`, `sample`, `Sample`, `orig.ident`, `batch`, or `donor`.

MERGE does not create sample groups by evenly splitting cells, and it does not replace missing biological annotations with clustering labels. Unresolved or ambiguous identifiers stop the workflow.

## Gene identifiers

Gene identifiers must be unique and use symbols compatible with the supplied GeneCompass ID/name mapping. During model construction, MERGE records the number and identity of unmapped genes. Unmapped input positions receive the padding token; they are never relabeled with numeric matrix indices.

For external evaluation, exact gene-symbol overlap determines the measured mask. A training gene absent from an external object receives an ignored label and a NaN response. It is not treated as zero expression, Down, or Unchanged.

## Descriptor identifiers

In `metal_class` mode, every biological sample from a condition shares the aggregated descriptor vector for that condition. Biological sample suffixes remain available for partitioning but do not create additional electrochemical measurements.

In `sample_id` mode, each descriptor row must have a unique `sample_id`, and every cell must map unambiguously to one row. Use this mode only when genuinely distinct descriptor vectors exist for those biological samples.

## External material manifest

```yaml
metal_feature_csv: /path/to/external_descriptor_replicates.csv
materials:
  - name: alloy_a
    h5ad: /path/to/alloy_a.h5ad
    feature_class: alloy_a
  - name: alloy_b
    h5ad: /path/to/alloy_b.h5ad
    feature_class: alloy_b
```

`name` is the evaluation label, `h5ad` points to the measured expression object, and `feature_class` selects replicate rows from the descriptor CSV. Names must be unique. No dataset-specific filenames are built into the evaluation code.
