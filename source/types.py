from typing import Literal

PretextTaskTypes = Literal["mae"]
DownstreamTaskTypes = Literal["classification"]

ViTBackbones = Literal[
    "vit_base_patch16",
    "vit_large_patch16",
    "vit_huge_patch14",
]
BackboneArchs = Literal[ViTBackbones]

### Datasets ###
DatasetsClassification = Literal[
    "dtd",
    "pets",
    "flowers",
    "aircrafts",
    "cub200",
    "voc07",
]

DownstreamDatasets = Literal[DatasetsClassification]
PretrainDatasets = Literal[
    "imagenet",
    "imagenet-1p",
    "imagenet-10p",
]

Datasets = Literal[PretrainDatasets, DownstreamDatasets]

DatasetSplits = Literal["unlabeled", "train", "val", "test", "trainval"]

OptimizerChoices = Literal["sgd", "adam", "adamw"]

RunTypes = Literal[
    "bissl",
    "bisslight",
    "ft-post-pretext",
    "ft-post-bissl",
    "ft-post-bisslight",
]

# BiSSL Specific
HinvSolverTypes = Literal[
    "cg_lw",
    "mfac_lw",
    # "mfac_global",
]
UpperLower = Literal["upper", "lower"]


BinaryChoices = Literal[0, 1]
FloatBetween0and1 = float  # The type checker will not enforce that the float is between 0 and 1, but this is meant as a hint to the user that the value should be in this range.
FloatNonNegative = float  # -II-
