# MERGE model definition

## Inputs

For each prediction instance, MERGE receives:

1. A normalized log1p expression vector from a control cell of the annotated cell type.
2. A five-dimensional material descriptor vector transformed and standardized with training-derived statistics.

The exposed-cell transcriptome is not an input. It is used only to construct labels when measured responses are available for training or evaluation.

## Cell encoder

Within the ordered model gene space, genes are ranked by positive expression. At most 2,048 genes are retained. Gene identifiers are mapped to the GeneCompass vocabulary, and expression values and an attention mask accompany the token IDs. GeneCompass Base produces a 768-dimensional representation from the first output token.

The pretrained backbone is jointly optimized with the new material and prediction modules unless specific lower layers are frozen through `genecompass.freeze_layers`.

## Material encoder

The canonical feature order is:

```text
Ecorr, icorr, Rp, f_MAX, Phase_MAX
```

Before standardization, `icorr`, `Rp`, and `f_MAX` undergo log10 transformation. The material network is:

```text
Linear(5, 128) → LayerNorm → GELU → Dropout
Linear(128, 256) → LayerNorm → GELU → Dropout
Linear(256, 768)
```

## Fusion and output

The cell and material embeddings are added element-wise. The prediction head is:

```text
Linear(768, 256) → LayerNorm → GELU → Dropout → Linear(256, 3G)
```

The output is reshaped to `(batch, G, 3)`. Class indices are fixed as:

```text
0 = Down
1 = Unchanged
2 = Up
```

## Response targets

For perturbation sample `s`, cell type `t`, and gene `g`:

```text
Δ(s,t,g) = [mean_log1p_perturbed(s,t,g) − mean_log1p_control(t,g)] / ln(2)
```

The target is Down when `Δ < −0.25`, Up when `Δ > 0.25`, and Unchanged otherwise. Groups below the configured minimum cell count receive `ignore_index` and do not enter loss or metric calculations.

## Optimization and selection

The reference configuration uses class-weighted cross entropy with label smoothing 0.05, AdamW, separate backbone/new-module learning rates, warmup followed by cosine decay, mixed precision on CUDA, gradient accumulation, and gradient clipping. Validation macro-F1 determines both the best checkpoint and early stopping.

## Group-level inference

A group is evaluated using multiple cell-type-matched control inputs. MERGE stores both majority-vote predictions and mean class probabilities. The number of sampled control cells must be reported for prospective prediction. External evaluation uses one sampled control input per retained exposed cell in each valid material–cell-type group.
