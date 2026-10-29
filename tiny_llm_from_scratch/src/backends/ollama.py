"""
Teacher Models and Pretrained Local Backends (Ollama & Local Integrations)
Provides strictly separated teacher/student distillation, critique, and local pretrained models.
Adheres to Non-Negotiable Rules:
- Detects installed models; never downloads large models without user consent
- Clearly labels which backend produced every answer
- Student from-scratch model remains independent and primary
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import urllib.error


@dataclass
class BackendInfo:
    backend_type: str  # "scratch", "pretrained_ollama", "teacher_distill"
    model_name: str
    available: bool
    status_message: str


class OllamaClient:
    """Safe, non-intrusive client for local Ollama server."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def is_server_running(self) -> bool:
        """Checks if local Ollama daemon is reachable with 1.0s timeout."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        """Lists locally downloaded models without initiating any downloads."""
        if not self.is_server_running():
            return []
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return sorted(models)
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Queries local Ollama model with strict labeling and timeout."""
        if not self.is_server_running():
            return {
                "success": False,
                "output": "",
                "error": "Ollama server is not running on http://localhost:11434.",
                "source": "ollama_unavailable",
            }

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                output_text = res.get("response", "").strip()
                return {
                    "success": True,
                    "output": output_text,
                    "model": model_name,
                    "source": f"teacher_model:ollama/{model_name}",
                    "eval_count": res.get("eval_count", 0),
                    "total_duration_sec": res.get("total_duration", 0) / 1e9,
                }
        except urllib.error.URLError as e:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to connect to Ollama: {e.reason}",
                "source": "ollama_connection_error",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "source": "ollama_error",
            }

    def generate_synthetic_instruction(
        self,
        domain: str,
        model_name: str,
        seed_concept: str = "",
    ) -> Optional[Dict[str, str]]:
        """
        Uses teacher model to formulate one high-quality training instruction pair.
        Strictly labeled as teacher-synthesized.
        """
        prompt = (
            f"You are creating a single high-quality question-answer training pair for an educational dataset.\n"
            f"Domain: {domain}\n"
            f"Specific topic: {seed_concept if seed_concept else 'key core principle'}\n"
            f"Format response STRICTLY as valid JSON with keys 'instruction' and 'response'. Do not add markdown backticks.\n"
            f"Example:\n{{\"instruction\": \"What is acceleration?\", \"response\": \"Acceleration is the rate of change of velocity per unit of time.\"}}"
        )
        res = self.generate(prompt, model_name=model_name, temperature=0.6, max_tokens=200)
        if not res.get("success"):
            return None

        txt = res.get("output", "").strip()
        # Clean potential markdown fences
        if txt.startswith("```json"):
            txt = txt[7:].strip()
        if txt.startswith("```"):
            txt = txt[3:].strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()

        try:
            pair = json.loads(txt)
            if "instruction" in pair and "response" in pair:
                pair["provenance"] = f"teacher_distillation:ollama/{model_name}"
                return pair
        except Exception:
            return None
        return None
