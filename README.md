# mlm-fine-tuning

Masked language-model pretraining and downstream fine-tuning for **INDUS-SDE**.

---

## INDUS-SDE — A Language Model for Scientific Content Curation and Discovery

This repository contains the **pretraining** (Weighted Dynamic Masking) and **downstream curation** code for INDUS-SDE, from our paper:

> **INDUS-SDE: A Language Model for Scientific Content Curation and Discovery**
> Pantha et al. — **KDD 2026, AI for Sciences Track** · DOI: [10.1145/3770855.3818847](https://doi.org/10.1145/3770855.3818847)

INDUS-SDE is a domain-adapted encoder pretrained with **Weighted Dynamic Masking (WDM)** on NASA's Science Discovery Engine (SDE) corpus. This repo covers (1) the MLM pretraining that produces the base encoder and (2) fine-tuning it for downstream curation tasks. The sentence transformer built on INDUS-SDE (**INDUS-SDE-ST**) lives in a separate repo: [`NASA-IMPACT/st-training-workflow`](https://github.com/NASA-IMPACT/st-training-workflow).

### Repository layout
| Path | Contents |
|---|---|
| [`mlm/`](mlm/) | MLM pretraining with Weighted Dynamic Masking — masking logic (`preprocess_data.py`), per-document YAKE keyword extraction (`keyword_extraction/`, `data_prep/generate_keyword*.py`), training entry point (`main.py`, `runn.sh`), configs (`config_*.json`), and MLM/masking analysis (`general_analysis/`, `embedding_analysis/`) |
| [`downstream_unification/`](downstream_unification/) | Multi-task fine-tuning + evaluation for the curation classifiers (GKR keyword recommendation, TDAMM, SMD division, content relevancy) — see its [README](downstream_unification/README.md) |

> **Branches:** `develop` carries the core WDM pretraining pipeline; **extended-context (1024-token) pretraining** — additional position embeddings (513–1024) initialized by copying the learned 1–512 embeddings, not randomly — lives on [`feature/indus-extended`](https://github.com/NASA-IMPACT/mlm-fine-tuning/tree/feature/indus-extended).

### Models (Hugging Face)
| Artifact | Model |
|---|---|
| Base encoder (INDUS-SDE) | [`nasa-impact/indus-sde-v0.2`](https://huggingface.co/nasa-impact/indus-sde-v0.2) |
| ModernBERT-SDE baseline | [`nasa-impact/modernbert-sde-v0.2`](https://huggingface.co/nasa-impact/modernbert-sde-v0.2) |
| GKR keyword classifier | [`nasa-impact/science-keyword-classification`](https://huggingface.co/nasa-impact/science-keyword-classification) |
| TDAMM classifier (INDUS-SDE) | [`nasa-impact/tdamm-classification-v2`](https://huggingface.co/nasa-impact/tdamm-classification-v2) |
| SMD division classifier | [`nasa-impact/division-classifier`](https://huggingface.co/nasa-impact/division-classifier) |
| SDE content relevancy | [`nasa-impact/sde-content-relevancy`](https://huggingface.co/nasa-impact/sde-content-relevancy) |

## Weighted Dynamic Masking (WDM)

WDM concentrates the MLM objective on scientific terminology. Each token is routed through one of two masking streams via an independent per-token draw: with probability **α = 0.85** it goes to **keyword masking** (selected only if YAKE flagged it a salient term), otherwise to **random masking** (selected regardless of content). Keywords are extracted **per document, on the fly** with YAKE — no corpus-wide precomputation.

- Masking implementation: [`mlm/preprocess_data.py`](mlm/preprocess_data.py) (static and dynamic keyword masking, IoU keyword matching, standard BERT 80/10/10 sub-strategy)
- Keyword extraction / extractor selection: [`mlm/keyword_extraction/`](mlm/keyword_extraction/), [`mlm/data_prep/`](mlm/data_prep/)

Key masking config (`mlm/config_indus_512_token_dkwm.json`):

```json
"keyword_masking": {
  "key_extrator": "YAKE",
  "mlm_probablity": 0.30,
  "keyword_masking_probablity": 0.85,
  "keyword_selection_percentile": 50.0,
  "keyword_iou_threshold": 0.80
},
"kw_masking_type": { "static": false, "dynamic": true }
```

## Running

MLM pretraining is driven by a JSON config:

```bash
cd mlm
python main.py --config_path config_indus_512_token_dkwm.json --wandb_mode online --gpu_index_to_use 0
```

Swap the config to change the masking regime (e.g. `config_indus.json`, `config_modernbert.json`). Downstream fine-tuning is configured via `downstream_unification/config.yaml` — see the [downstream README](downstream_unification/README.md).

## Citation

If you use INDUS-SDE or this code in your research, please cite:

```bibtex
@inproceedings{pantha2026indussde,
  author    = {Pantha, Nishan and Awale, Sajil and Kuruvanthodi, Vishnudev and KC, Simran and Ramasubramanian, Muthukumaran and Davis, Carson and Praveen, Bishwas and Foshee, Emily and Bhattacharjee, Bishwaranjan and Bugbee, Kaylin and Ramachandran, Rahul},
  title     = {{INDUS-SDE}: A Language Model for Scientific Content Curation and Discovery},
  year      = {2026},
  isbn      = {979-8-4007-2259-2},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3770855.3818847},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)},
  location  = {Jeju Island, Republic of Korea},
  series    = {KDD '26}
}
```
</content>
