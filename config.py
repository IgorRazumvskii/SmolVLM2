import os
from pathlib import Path
import torch


class Config:
    
    def __init__(self):
        # Режим работы: cuda, cpu, mps, auto
        device_env = os.getenv("DEVICE", "cpu")
        
        # Размер модели для SmolVLM2: '256M', '500M', '2.2B' (или старые '1b', '2b')
        self.model_size = os.getenv("MODEL_SIZE", "256M")
        
        # Порт приложения
        self.port = int(os.getenv("PORT", "7860"))
        
        self.models_dir = Path(os.getenv("MODELS_DIR", "/app/models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.hf_cache_dir = Path(os.getenv("HF_HOME", "/app/models/hf_cache"))
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = os.getenv("MODEL_NAME", f"smolvlm2-{self.model_size}")
        self.hf_model_id = os.getenv("HF_MODEL_ID", None)
        
        if device_env == "auto" and torch is not None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device_env

        os.environ["HF_HOME"] = str(self.hf_cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.hf_cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.hf_cache_dir / "datasets")
        

