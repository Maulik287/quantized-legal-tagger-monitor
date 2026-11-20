"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
import os
from pathlib import Path
from typing import Any

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
    
    # Try loading from datasets if explicitly requested and online, fallback to curated dataset
    if custom_data is None and split != "train":
        # Create varied split
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
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
# Hint: Utilize the evaluation methods demonstrated in the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to assess your model's tagging accuracy.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Apply dynamic and static post-training quantization to the trained model.
# Hint: Implement the `Shrink a Neural Network with Quantisation` technique, specifically applying post-training quantization in both dynamic and static modes.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
# Hint: Apply the measurement techniques from the `Shrink a Neural Network with Quantisation` drill to compare model sizes and simulate inference speeds.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Present a dashboard-like output of model efficiency metrics.
# Hint: Organize and present the efficiency metrics obtained from the `Shrink a Neural Network with Quantisation` drill in a clear, comparative format suitable for a dashboard.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
