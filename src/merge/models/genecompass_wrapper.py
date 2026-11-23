"""Strict GeneCompass Base adapter for control-cell expression encoding."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import nn


class GeneCompassWrapper(nn.Module):
    """Load pretrained GeneCompass and encode ranked control-cell expression.

    Gene symbols that cannot be mapped to the GeneCompass vocabulary are padded
    and reported explicitly. Numeric array indices are never used as substitutes
    for GeneCompass tokens.
    """

    def __init__(
        self,
        pretrained_dir: str,
        token_dict_path: str,
        prior_knowledge_dir: str,
        freeze_layers: int = 0,
        hidden_dim: int = 768,
        max_tokens: int = 2048,
        gene_names: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if hidden_dim != 768:
            raise ValueError("MERGE uses GeneCompass Base with hidden_dim=768")
        if gene_names is None or len(gene_names) == 0:
            raise ValueError("The ordered training gene names are required")
        self.hidden_dim = int(hidden_dim)
        self.max_tokens = int(max_tokens)
        self.gene_names = [str(gene) for gene in gene_names]
        self.logger = logger or logging.getLogger(__name__)
        self.token_dict_path = str(Path(token_dict_path).expanduser())
        self.prior_knowledge_dir = str(Path(prior_knowledge_dir).expanduser())
        self.token_dict = self._load_pickle(self.token_dict_path, "token dictionary")
        if not isinstance(self.token_dict, dict) or not self.token_dict:
            raise ValueError("The GeneCompass token dictionary must be a non-empty dictionary")
        self.gene_name_to_token = self._load_gene_name_to_token()
        gene_index_to_token = self._build_gene_index_to_token()
        self.register_buffer("gene_index_to_token", gene_index_to_token, persistent=False)
        self.encoder = self._build_genecompass_encoder(Path(pretrained_dir).expanduser())
        encoder_dim = int(getattr(getattr(self.encoder, "config", None), "hidden_size", hidden_dim))
        if encoder_dim != self.hidden_dim:
            raise ValueError(f"GeneCompass hidden size {encoder_dim} does not match configured {self.hidden_dim}")
        self._apply_freezing(int(freeze_layers))

    @staticmethod
    def _load_pickle(path: str, description: str):
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"GeneCompass {description} not found: {file_path}")
        with file_path.open("rb") as handle:
            return pickle.load(handle)

    def _load_gene_name_to_token(self) -> Dict[str, int]:
        mapping_path = (
            Path(self.prior_knowledge_dir)
            / "gene_list"
            / "Gene_id_name_dict_human_mouse.pickle"
        )
        id_to_name = self._load_pickle(str(mapping_path), "gene ID/name mapping")
        mapping: Dict[str, int] = {}
        for gene_id, gene_name in id_to_name.items():
            token = self.token_dict.get(gene_id)
            if token is None:
                continue
            if isinstance(token, (tuple, list)):
                token = token[0] if token else None
            try:
                token = int(token)
            except (TypeError, ValueError):
                continue
            for key in {str(gene_name), str(gene_name).upper()}:
                mapping.setdefault(key, token)
        return mapping

    def _build_gene_index_to_token(self) -> torch.Tensor:
        token_ids = []
        unmapped = []
        for gene in self.gene_names:
            token = self.token_dict.get(gene, self.token_dict.get(gene.upper()))
            if token is None:
                token = self.gene_name_to_token.get(
                    gene, self.gene_name_to_token.get(gene.upper())
                )
            if isinstance(token, (tuple, list)):
                token = token[0] if token else None
            try:
                token_ids.append(int(token))
            except (TypeError, ValueError):
                token_ids.append(0)
                unmapped.append(gene)
        self.unmapped_gene_names = unmapped
        self.gene_mapping_coverage = 1.0 - len(unmapped) / len(self.gene_names)
        if len(unmapped) == len(self.gene_names):
            raise ValueError("No input genes map to GeneCompass tokens")
        if unmapped:
            self.logger.warning(
                "Mapped %d/%d genes to GeneCompass tokens; %d unmapped genes use padding",
                len(self.gene_names) - len(unmapped),
                len(self.gene_names),
                len(unmapped),
            )
        return torch.tensor(token_ids, dtype=torch.long)

    def _build_genecompass_encoder(self, pretrained_dir: Path) -> nn.Module:
        if not pretrained_dir.is_dir():
            raise FileNotFoundError(f"GeneCompass Base checkpoint directory not found: {pretrained_dir}")
        try:
            from genecompass import BertForMaskedLM
            from genecompass.utils import load_prior_embedding
        except ImportError as error:
            raise ImportError(
                "GeneCompass is not installed. Install the official package from "
                "https://github.com/xCompass-AI/GeneCompass before building MERGE."
            ) from error

        prior_dir = Path(self.prior_knowledge_dir)
        relative_paths = [
            "promoter_emb/human_emb_768.pickle",
            "promoter_emb/mouse_emb_768.pickle",
            "gene_co_express_emb/Human_dim_768_gene_28291_random.pickle",
            "gene_co_express_emb/Mouse_dim_768_gene_27444_random.pickle",
            "gene_family/Human_dim_768_gene_28291_random.pickle",
            "gene_family/Mouse_dim_768_gene_27934_random.pickle",
            "PECA2vec/human_PECA_vec.pickle",
            "PECA2vec/mouse_PECA_vec.pickle",
            "gene_list/Gene_id_name_dict_human_mouse.pickle",
            "homologous_hm_token.pickle",
        ]
        required = [prior_dir / relative for relative in relative_paths]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing GeneCompass prior-knowledge files: " + ", ".join(missing))
        embeddings = load_prior_embedding(
            name2promoter_human_path=str(required[0]),
            name2promoter_mouse_path=str(required[1]),
            name2coexp_human_path=str(required[2]),
            name2coexp_mouse_path=str(required[3]),
            name2family_human_path=str(required[4]),
            name2family_mouse_path=str(required[5]),
            name2peca_human_path=str(required[6]),
            name2peca_mouse_path=str(required[7]),
            id2name_human_mouse_path=str(required[8]),
            token_dictionary_or_path=self.token_dict_path,
            homologous_gene_path=str(required[9]),
        )
        knowledges = {
            "promoter": embeddings[0].float(),
            "co_exp": embeddings[1].float(),
            "gene_family": embeddings[2].float(),
            "peca_grn": embeddings[3].float(),
            "homologous_gene_human2mouse": embeddings[4],
        }
        model = BertForMaskedLM.from_pretrained(str(pretrained_dir), knowledges=knowledges)
        return model.bert

    def _apply_freezing(self, freeze_layers: int) -> None:
        if freeze_layers < 0:
            raise ValueError("freeze_layers must be non-negative")
        if freeze_layers == 0:
            return
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            raise ValueError("Unable to locate GeneCompass Transformer layers for freezing")
        if freeze_layers > len(layers):
            raise ValueError(f"freeze_layers={freeze_layers} exceeds available layers={len(layers)}")
        for layer in list(layers)[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def _rank_to_inputs(self, expression: torch.Tensor):
        if expression.ndim != 2 or expression.shape[1] != len(self.gene_names):
            raise ValueError(
                f"Expected expression with shape (batch, {len(self.gene_names)}), "
                f"got {tuple(expression.shape)}"
            )
        top_k = min(self.max_tokens, expression.shape[1])
        values, indices = torch.topk(expression, k=top_k, dim=1, largest=True, sorted=True)
        token_ids = self.gene_index_to_token[indices]
        positive = values > 0
        token_ids = torch.where(positive, token_ids.long(), torch.zeros_like(token_ids))
        values = torch.where(positive, values.float(), torch.zeros_like(values, dtype=torch.float32))
        attention_mask = (token_ids != 0).long()
        return token_ids, values, attention_mask

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """Return one 768-dimensional embedding per control-cell profile."""
        token_ids, values, attention_mask = self._rank_to_inputs(expression)
        species = torch.zeros(
            (token_ids.shape[0], 1), dtype=torch.long, device=token_ids.device
        )
        output = self.encoder(
            input_ids=token_ids,
            values=values,
            attention_mask=attention_mask,
            species=species,
        )
        if isinstance(output, torch.Tensor):
            hidden = output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            raise RuntimeError("Unsupported GeneCompass output type")
        return hidden[:, 0, :] if hidden.ndim == 3 else hidden
