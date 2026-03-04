from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class KGConfig(BaseModel):
    llm_provider: str = "mock"  # "openai" | "mock"
    openai_model: str = "gpt-4.1-mini"  # placeholder
    taxonomy_cso_url: Optional[str] = None
    normalize_with_wordnet: bool = True

class ModelConfig(BaseModel):
    text_model_name: str = "bert-base-uncased"
    code_model_name: str = "microsoft/codebert-base"
    text_dim: int = 768
    code_dim: int = 768
    student_profile_dim: int = 32
    student_lstm_hidden: int = 128
    student_state_dim: int = 256  # last hidden + profile features (paper)
    concept_dim_in: int = 100
    rgcn_hidden: int = 256
    fused_dim: int = 256
    dropout: float = 0.2

class TrainConfig(BaseModel):
    seed: int = 42
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 20
    patience: int = 3
    grad_clip_norm: float = 1.0
    device: str = "cuda"  # "cpu" | "cuda"
    output_dir: str = "outputs/run"

class DataConfig(BaseModel):
    problems_path: str
    interactions_path: str
    user_profiles_path: Optional[str] = None
    kg_dir: Optional[str] = None

class AppConfig(BaseModel):
    data: DataConfig
    kg: KGConfig = Field(default_factory=KGConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    extra: Dict[str, Any] = Field(default_factory=dict)
