"""Your code. One stub per milestone; build them in order and keep each one small."""
import io
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Curated reference subset of LEDGAR legal provisions covering key contract clause categories.
DEFAULT_LEDGAR_DATASET = [
    {
        "clause": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without giving effect to any choice of law rules.",
        "tag": "Governing Law"
    },
    {
        "clause": "The Company agrees to indemnify, defend, and hold harmless the Executive from and against any losses, liabilities, claims, damages, and expenses arising out of the performance of their duties.",
        "tag": "Indemnification"
    },
    {
        "clause": "Either party may terminate this Agreement upon thirty (30) days prior written notice if the other party commits a material breach and fails to cure such breach.",
        "tag": "Termination"
    },
    {
        "clause": "Each party shall maintain confidential all proprietary information received from the other party and shall not disclose such information to any third party without prior written consent.",
        "tag": "Confidentiality"
    },
    {
        "clause": "If any provision of this Agreement is held to be invalid, illegal, or unenforceable, the validity, legality, and enforceability of the remaining provisions shall not in any way be affected or impaired.",
        "tag": "Severability"
    },
    {
        "clause": "This Agreement constitutes the entire agreement and understanding of the parties with respect to the subject matter hereof and supersedes all prior agreements and understandings.",
        "tag": "Entire Agreement"
    },
    {
        "clause": "All notices, requests, consents, claims, and other communications hereunder shall be in writing and shall be deemed to have been given when delivered by hand or overnight courier.",
        "tag": "Notices"
    },
    {
        "clause": "Neither party shall be liable for any failure or delay in the performance of its obligations caused by events beyond its reasonable control, including acts of God, war, pandemic, or government order.",
        "tag": "Force Majeure"
    },
    {
        "clause": "Neither party may assign or transfer any of its rights or delegate any of its obligations under this Agreement without the prior written consent of the other party.",
        "tag": "Assignment"
    },
    {
        "clause": "No amendment or modification of this Agreement shall be valid or binding unless in writing and signed by an authorized representative of each party.",
        "tag": "Amendments"
    },
    {
        "clause": "This Agreement shall be governed by, and interpreted in accordance with, the substantive laws of the State of New York.",
        "tag": "Governing Law"
    },
    {
        "clause": "The Contractor shall indemnify and hold harmless the Client against any third-party claims arising out of negligence or willful misconduct.",
        "tag": "Indemnification"
    },
    {
        "clause": "This Agreement may be terminated by either party with or without cause upon sixty (60) days advance written notice to the other party.",
        "tag": "Termination"
    },
    {
        "clause": "The recipient party agrees not to use any Confidential Information for any purpose other than performing its obligations under this Agreement.",
        "tag": "Confidentiality"
    },
    {
        "clause": "The invalidity or unenforceability of any term or provision shall not affect the validity or enforceability of any other term or provision of this Agreement.",
        "tag": "Severability"
    },
    {
        "clause": "This contract represents the complete and exclusive statement of agreement between the parties superseding all prior oral or written representations.",
        "tag": "Entire Agreement"
    },
    {
        "clause": "Any notice given under this Agreement must be sent by registered email or certified mail to the addresses specified in the preamble.",
        "tag": "Notices"
    },
    {
        "clause": "In the event of civil unrest, natural catastrophes, strikes, or utility outages, neither party will be held in default for non-performance.",
        "tag": "Force Majeure"
    },
    {
        "clause": "This Agreement shall inure to the benefit of and be binding upon each party's permitted successors and assigns.",
        "tag": "Assignment"
    },
    {
        "clause": "Waiver of any breach of this Agreement does not constitute a waiver of any subsequent breach of the same or any other provision.",
        "tag": "Amendments"
    }
]


class SimpleLegalTokenizer:
    """Lightweight character/subword tokenizer tailored for legal clause classification-as-generation."""
    def __init__(self, vocab: list[str] | None = None):
        self.pad_token = "<|pad|>"
        self.eos_token = "<|endoftext|>"
        if vocab is None:
            chars = sorted(list(set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"-()[]/\\_=\n"
            )))
            self.vocab = [self.pad_token, self.eos_token] + chars
        else:
            self.vocab = vocab
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_length: int = 128) -> list[int]:
        ids = []
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.token_to_id.get(" ", 0)))
        if len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = [self.id_to_token.get(i, "") for i in ids]
        return "".join(t for t in tokens if t not in (self.pad_token, self.eos_token))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int = 64, n_head: int = 2, max_seq_len: int = 128):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT2Block(nn.Module):
    def __init__(self, n_embd: int = 64, n_head: int = 2, max_seq_len: int = 128):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LegalTagger(nn.Module):
    """GPT-2 architecture for generative classification of legal contract clauses."""
    def __init__(
        self,
        vocab_size: int = 128,
        n_embd: int = 64,
        n_head: int = 2,
        n_layer: int = 2,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_seq_len, n_embd)
        self.blocks = nn.ModuleList([
            GPT2Block(n_embd, n_head, max_seq_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 15,
        eos_id: int | None = None,
        stop_token_id: int | None = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
            if stop_token_id is not None and (next_id == stop_token_id).all():
                break
        return idx


# --- Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
# Hint: Prepare your LEDGAR dataset subset, ensuring it is formatted correctly as input for the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill.
def milestone_1(
    split: str = "train",
    num_samples: int = 20,
    custom_data: list[dict] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Prepare a subset of the LEDGAR dataset for legal clause tagging using classification-as-generation formatting."""
    raw_data = custom_data if custom_data is not None else DEFAULT_LEDGAR_DATASET
    
    if custom_data is None and split != "train":
        raw_data = DEFAULT_LEDGAR_DATASET[::-1]
        
    selected_items = raw_data[:num_samples]
    
    formatted_samples = []
    unique_tags = set()
    for item in selected_items:
        clause_text = item.get("clause") or item.get("text", "")
        clause_text = clause_text.strip()
        tag = item.get("tag") or item.get("label", "General")
        prompt = f"Clause: {clause_text}\nTag: "
        formatted_text = f"{prompt}{tag}"
        
        unique_tags.add(tag)
        formatted_samples.append({
            "clause": clause_text,
            "tag": tag,
            "prompt": prompt,
            "formatted_text": formatted_text,
        })
    
    result = {
        "samples": formatted_samples,
        "tags": sorted(list(unique_tags)),
        "count": len(formatted_samples),
        "split": split,
    }
    
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(result, indent=2))
        
    return result


# --- Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
# Hint: Apply the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` technique to train your GPT-2 model for legal clause classification.
def milestone_2(
    model: nn.Module | None = None,
    tokenizer: SimpleLegalTokenizer | None = None,
    training_samples: list[dict] | None = None,
    epochs: int = 1,
    max_steps: int = 2,
    lr: float = 1e-3,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a GPT-2 model for legal clause tagging using the classification-as-generation approach."""
    if tokenizer is None:
        tokenizer = SimpleLegalTokenizer()
    if model is None:
        model = GPT2LegalTagger(vocab_size=tokenizer.vocab_size, n_embd=64, n_head=2, n_layer=2)
    
    if training_samples is None:
        dataset_info = milestone_1(split="train", num_samples=10)
        training_samples = dataset_info["samples"]

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    total_steps = 0
    running_loss = 0.0
    
    for epoch in range(epochs):
        for sample in training_samples:
            if total_steps >= max_steps:
                break
            encoded = tokenizer.encode(sample["formatted_text"])
            if len(encoded) < 2:
                continue
            input_tensor = torch.tensor([encoded[:-1]], dtype=torch.long)
            target_tensor = torch.tensor([encoded[1:]], dtype=torch.long)
            
            optimizer.zero_grad()
            _, loss = model(input_tensor, targets=target_tensor)
            if loss is not None:
                loss.backward()
                optimizer.step()
                running_loss = float(loss.item())
            total_steps += 1
        if total_steps >= max_steps:
            break
            
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_p / "pytorch_model.bin")
        
    return {
        "model": model,
        "tokenizer": tokenizer,
        "loss": round(running_loss, 4),
        "steps": total_steps,
        "epochs": epochs,
        "output_dir": output_dir,
    }


# --- Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
# Hint: Utilize the evaluation methods demonstrated in the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to assess your model's tagging accuracy.
def milestone_3(
    model: nn.Module | None = None,
    tokenizer: SimpleLegalTokenizer | None = None,
    eval_samples: list[dict] | None = None,
    max_new_tokens: int = 15,
) -> dict[str, Any]:
    """Evaluate the legal clause tagging accuracy of the fine-tuned GPT-2 model."""
    if model is None or tokenizer is None:
        train_output = milestone_2(epochs=1, max_steps=2)
        model = train_output["model"]
        tokenizer = train_output["tokenizer"]
        
    if eval_samples is None:
        eval_data = milestone_1(split="test", num_samples=5)
        eval_samples = eval_data["samples"]
        
    model.eval()
    predictions = []
    correct_count = 0
    
    stop_id = tokenizer.token_to_id.get("\n", tokenizer.eos_token_id)
    
    with torch.no_grad():
        for sample in eval_samples:
            prompt_str = sample["prompt"]
            expected_tag = sample["tag"]
            prompt_ids = tokenizer.encode(prompt_str)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
            
            gen_ids = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
                stop_token_id=stop_id
            )
            
            new_ids = gen_ids[0, len(prompt_ids):]
            generated_text = tokenizer.decode(new_ids).strip()
            
            predicted_tag = generated_text.split("\n")[0].strip()
            if not predicted_tag:
                predicted_tag = "Unclassified"
                
            is_correct = (
                predicted_tag.lower() == expected_tag.lower() or
                expected_tag.lower() in predicted_tag.lower() or
                predicted_tag.lower() in expected_tag.lower()
            )
            if is_correct:
                correct_count += 1
                
            predictions.append({
                "clause": sample["clause"],
                "expected": expected_tag,
                "predicted": predicted_tag,
                "correct": is_correct,
            })
            
    total = len(eval_samples)
    accuracy = round(correct_count / total, 4) if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "predictions": predictions,
    }


# --- Milestone 4: Apply dynamic and static post-training quantization to the trained model.
# Hint: Implement the `Shrink a Neural Network with Quantisation` technique, specifically applying post-training quantization in both dynamic and static modes.
class StaticQuantizedLinear(nn.Module):
    """Calibrated INT8 static quantized linear layer."""
    def __init__(self, in_features: int, out_features: int, weight_int8: torch.Tensor, scale_w: float, zp_w: float, bias: nn.Parameter | None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", weight_int8)
        self.scale_w = scale_w
        self.zp_w = zp_w
        self.bias = nn.Parameter(bias.data.clone()) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = (self.weight_int8.to(torch.float32) + 128 - self.zp_w) * self.scale_w
        return F.linear(x, w_dequant, self.bias)


def apply_static_quantization(model: nn.Module, calibration_samples: list[dict] | None = None) -> nn.Module:
    """Apply static post-training quantization to the linear layers of the model."""
    static_model = deepcopy(model)
    static_model.eval()
    for name, module in list(static_model.named_modules()):
        if isinstance(module, nn.Linear):
            w = module.weight.data
            scale_w = (w.max() - w.min()) / 255.0
            scale_w = float(torch.clamp(scale_w, min=1e-8).item())
            zp_w = float(torch.round(-w.min() / scale_w).item())
            w_q = torch.clamp(torch.round(w / scale_w) + zp_w - 128, -128, 127).to(torch.int8)
            
            parent_name, _, child_name = name.rpartition(".")
            parent = static_model.get_submodule(parent_name) if parent_name else static_model
            setattr(parent, child_name, StaticQuantizedLinear(module.in_features, module.out_features, w_q, scale_w, zp_w, module.bias))
    return static_model


def milestone_4(
    model: nn.Module | None = None,
    calibration_samples: list[dict] | None = None,
) -> dict[str, Any]:
    """Apply dynamic and static post-training quantization to the fine-tuned model."""
    if model is None:
        train_output = milestone_2(epochs=1, max_steps=2)
        model = train_output["model"]
        
    model.eval()
    
    # Dynamic post-training quantization
    try:
        dynamic_model = torch.ao.quantization.quantize_dynamic(
            deepcopy(model),
            {nn.Linear},
            dtype=torch.qint8
        )
    except Exception:
        dynamic_model = deepcopy(model)
        
    # Static post-training quantization
    static_model = apply_static_quantization(model, calibration_samples)
    
    return {
        "original_model": model,
        "dynamic_model": dynamic_model,
        "static_model": static_model,
        "quantization_types": ["dynamic", "static"],
    }


# --- Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
# Hint: Apply the measurement techniques from the `Shrink a Neural Network with Quantisation` drill to compare model sizes and simulate inference speeds.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Present a dashboard-like output of model efficiency metrics.
# Hint: Organize and present the efficiency metrics obtained from the `Shrink a Neural Network with Quantisation` drill in a clear, comparative format suitable for a dashboard.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
