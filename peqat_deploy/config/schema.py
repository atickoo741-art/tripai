from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExportJob:
    source_dir: str
    source_basename: str
    selected_checkpoint: Optional[str]
    export_dir: str
    base_model_name: str
    weight_bits: int = 4
    activation_bits: int = 4
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    block_size: int = 2048
    calibration_percentile: float = 99.5
    learn_smooth: bool = True
    calibrate_smooth: bool = True
    dtype: str = "float16"
    save_validation: bool = False
    overwrite: bool = False
