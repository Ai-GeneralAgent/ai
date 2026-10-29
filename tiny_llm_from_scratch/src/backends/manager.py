"""
Unified Backend Manager
Manages from-scratch checkpoints, optional local Ollama pretrained models,
and teacher distillation modes while enforcing strict provenance labeling.
"""
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from .ollama import OllamaClient, BackendInfo
from ..models.transformer import ModernTransformer
from ..models.checkpoint import load_modern_checkpoint, inspect_checkpoint
from ..generation.sampler import TextGenerator, SamplingConfig


class BackendSource(Enum):
    SCRATCH_MODEL = "scratch_from_scratch"
    PRETRAINED_OLLAMA = "pretrained_local_ollama"
    TEACHER_DISTILL = "teacher_distillation"


class BackendManager:
    """Coordinates generation backends and guarantees transparent origin attribution."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        ollama_host: str = "http://localhost:11434",
    ):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.ollama = OllamaClient(host=ollama_host)
        self.active_source: BackendSource = BackendSource.SCRATCH_MODEL
        self.active_scratch_model: Optional[ModernTransformer] = None
        self.active_scratch_path: Optional[Path] = None
        self.active_ollama_model: Optional[str] = None

    def list_available_backends(self) -> List[BackendInfo]:
        """Audits both scratch checkpoints and local Ollama models."""
        backends: List[BackendInfo] = []

        # 1. Scratch checkpoints
        ckpt_dir = self.project_root / "checkpoints"
        if ckpt_dir.is_dir():
            pts = sorted(ckpt_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
            for p in pts:
                try:
                    info = inspect_checkpoint(p)
                    backends.append(BackendInfo(
                        backend_type="scratch",
                        model_name=p.name,
                        available=True,
                        status_message=f"From-Scratch {info.get('format')} (step {info.get('step')})",
                    ))
                except Exception as e:
                    backends.append(BackendInfo(
                        backend_type="scratch",
                        model_name=p.name,
                        available=False,
                        status_message=f"Unloadable: {e}",
                    ))

        # 2. Local Ollama server & models
        if self.ollama.is_server_running():
            models = self.ollama.list_installed_models()
            for m in models:
                backends.append(BackendInfo(
                    backend_type="pretrained_ollama",
                    model_name=m,
                    available=True,
                    status_message="Installed local Ollama model",
                ))
        else:
            backends.append(BackendInfo(
                backend_type="pretrained_ollama",
                model_name="none",
                available=False,
                status_message="Local Ollama daemon not reachable on port 11434",
            ))

        return backends

    def set_scratch_model(self, model: ModernTransformer, ckpt_path: Path):
        self.active_scratch_model = model
        self.active_scratch_path = ckpt_path
        self.active_source = BackendSource.SCRATCH_MODEL

    def set_ollama_model(self, model_name: str) -> bool:
        installed = self.ollama.list_installed_models()
        if model_name in installed:
            self.active_ollama_model = model_name
            self.active_source = BackendSource.PRETRAINED_OLLAMA
            return True
        return False

    def generate(
        self,
        prompt: str,
        tokenizer: Any,
        sampling_config: Optional[SamplingConfig] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Executes generation using the currently selected backend.
        Never disguises or mixes origins.
        """
        cfg = sampling_config or SamplingConfig()

        if self.active_source == BackendSource.PRETRAINED_OLLAMA and self.active_ollama_model:
            res = self.ollama.generate(
                prompt=prompt,
                model_name=self.active_ollama_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )
            return {
                "output": res.get("output", ""),
                "backend_type": "pretrained_local_ollama",
                "backend_model": self.active_ollama_model,
                "label": f"[Source: Pretrained Ollama ({self.active_ollama_model})]",
                "raw_result": res,
            }

        # Default: From-scratch model
        if self.active_scratch_model and tokenizer:
            gen = TextGenerator(self.active_scratch_model, tokenizer)
            out = gen.generate(prompt, config=cfg, device=device)
            ckpt_name = self.active_scratch_path.name if self.active_scratch_path else "in-memory"
            return {
                "output": out,
                "backend_type": "scratch_from_scratch",
                "backend_model": ckpt_name,
                "label": f"[Source: From-Scratch ModernTransformer ({ckpt_name})]",
            }

        return {
            "output": "No active model backend is currently loaded.",
            "backend_type": "none",
            "backend_model": "none",
            "label": "[Source: Error - No Backend]",
        }
