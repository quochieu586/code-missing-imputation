"""Schouten-only synthetic masking; latent-p MNAR and random blocks are NOT-READY."""
from .capabilities import NotReadyError, capabilities, require_capability
from .masking import MaskResult, mask_counts, mask_fixed_wave
from .reference import reproduce_table1
from .generator import GeneratorConfig, generate, export_dataset
from .datasets import load_conditioning, load_evaluation

__all__ = ["NotReadyError", "capabilities", "require_capability", "MaskResult",
           "mask_counts", "mask_fixed_wave", "reproduce_table1", "GeneratorConfig",
           "generate", "export_dataset", "load_conditioning", "load_evaluation"]
