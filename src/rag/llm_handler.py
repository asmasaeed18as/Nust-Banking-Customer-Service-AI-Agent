"""
src/rag/llm_handler.py
───────────────────────
LLM interface supporting two provider modes:

  Mode 1 — "hf_api"  : HuggingFace Inference API (cloud, no GPU needed)
  Mode 2 — "local"   : Local HuggingFace model via transformers + bitsandbytes

Controlled via config/settings.py → LLM_PROVIDER.

Prompt format: Qwen2.5-Instruct / Llama chat template
  <|im_start|>system\n{system}<|im_end|>
  <|im_start|>user\n{user}<|im_end|>
  <|im_start|>assistant\n
"""

import os
import time
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import (
    LLM_PROVIDER, HF_MODEL_ID, HF_API_TOKEN,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
)


class LLMHandler:
    """
    Unified LLM handler. Initialises once, inference on every call.

    Usage:
        handler = LLMHandler()
        answer  = handler.generate(system_prompt, user_prompt)
    """

    def __init__(self):
        logger.info(f"[LLMHandler] Initialising LLM. Provider: {LLM_PROVIDER} | Model: {HF_MODEL_ID}")

        if LLM_PROVIDER == "hf_api":
            self._init_hf_api()
        elif LLM_PROVIDER == "local":
            self._init_local()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Choose 'hf_api' or 'local'.")

    # ── HuggingFace Inference API ─────────────────────────────────────────────
    def _init_hf_api(self):
        from huggingface_hub import InferenceClient
        token = HF_API_TOKEN or os.getenv("HF_API_TOKEN")
        if not token:
            logger.warning(
                "[LLMHandler] HF_API_TOKEN not set. "
                "Requests will be rate-limited (free anonymous tier). "
                "Set HF_API_TOKEN in your .env for higher limits."
            )
        self.client = InferenceClient(
            model=HF_MODEL_ID,
            token=token or None,
        )
        logger.success(f"[LLMHandler] HuggingFace API client ready → {HF_MODEL_ID}")
        self._mode = "hf_api"


    # ── Local transformers model ───────────────────────────────────────────────
    def _init_local(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"[LLMHandler] Loading local model: {HF_MODEL_ID} ...")
        logger.info(f"[LLMHandler] Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16,   # ~6 GB VRAM for 3B — fits RTX 3080 (10 GB)
            device_map="auto",           # sends to CUDA automatically
            trust_remote_code=True,
        )
        self.model.eval()

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            logger.success(f"[LLMHandler] Local model loaded on GPU. VRAM used: {vram_used:.1f} GB")
        else:
            logger.success(f"[LLMHandler] Local model loaded on CPU.")
        self._mode = "local"

    # ── Generate ──────────────────────────────────────────────────────────────
    def generate(self, system: str, user: str) -> str:
        """
        Generate an LLM response given a system prompt and user message.

        Returns:
            The generated text string.
        """
        logger.info(f"[LLMHandler] Generating response. System len={len(system)}, User len={len(user)}")
        t0 = time.perf_counter()

        if self._mode == "hf_api":
            response = self._generate_hf_api(system, user)
        else:
            response = self._generate_local(system, user)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.success(f"[LLMHandler] Generation complete in {elapsed:.0f} ms. Output tokens ≈ {len(response.split())}")
        return response.strip()

    def _generate_hf_api(self, system: str, user: str) -> str:
        """
        Calls the HuggingFace chat_completion endpoint.
        Qwen2.5-Instruct and Llama-3 both support the messages format.
        """
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
        ]
        try:
            completion = self.client.chat_completion(
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return completion.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "unauthorized" in err or "token" in err:
                logger.error("[LLMHandler] HF API Auth failed — HF_API_TOKEN is missing or invalid in .env")
                return (
                    "[LLM AUTH ERROR] HuggingFace API token is missing or invalid. "
                    "Please add your HF_API_TOKEN to the .env file.\n"
                    "Get a free token at: https://huggingface.co/settings/tokens"
                )
            elif "429" in err or "rate limit" in err:
                logger.warning("[LLMHandler] HF API rate limit hit.")
                return (
                    "I'm receiving too many requests right now. "
                    "Please wait a moment and try again."
                )
            else:
                logger.error(f"[LLMHandler] HF API call failed: {e}")
                return (
                    "I'm sorry, I'm experiencing a temporary issue connecting to the AI service. "
                    "Please try again in a moment, or contact our support team for immediate assistance."
                )

    def _generate_local(self, system: str, user: str) -> str:
        """
        Runs inference on the locally loaded model using the Qwen chat template.
        """
        import torch
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
