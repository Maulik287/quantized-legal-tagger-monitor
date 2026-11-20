"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)

# Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def test_milestone_1():
    data = milestone_1(num_samples=10)
    assert isinstance(data, dict)
    assert "samples" in data
    assert "tags" in data
    assert "count" in data
    assert data["count"] == 10
    assert len(data["samples"]) == 10
    sample = data["samples"][0]
    assert "clause" in sample
    assert "tag" in sample
    assert "prompt" in sample
    assert "formatted_text" in sample
    assert sample["prompt"].startswith("Clause: ")
    assert sample["prompt"].endswith("\nTag: ")
    assert sample["formatted_text"].startswith(sample["prompt"])
    assert sample["tag"] in data["tags"]


# Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
def test_milestone_2():
    train_res = milestone_2(epochs=1, max_steps=2)
    assert isinstance(train_res, dict)
    assert "model" in train_res
    assert "tokenizer" in train_res
    assert "loss" in train_res
    assert "steps" in train_res
    assert train_res["loss"] >= 0.0
    assert train_res["steps"] >= 1


# Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def test_milestone_3():
    eval_res = milestone_3()
    assert isinstance(eval_res, dict)
    assert "accuracy" in eval_res
    assert "correct" in eval_res
    assert "total" in eval_res
    assert "predictions" in eval_res
    assert eval_res["total"] > 0
    assert 0.0 <= eval_res["accuracy"] <= 1.0
    assert len(eval_res["predictions"]) == eval_res["total"]
    pred = eval_res["predictions"][0]
    assert "expected" in pred
    assert "predicted" in pred
    assert "correct" in pred


# Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def test_milestone_4():
    quant_res = milestone_4()
    assert isinstance(quant_res, dict)
    assert "original_model" in quant_res
    assert "dynamic_model" in quant_res
    assert "static_model" in quant_res
    assert "quantization_types" in quant_res
    assert "dynamic" in quant_res["quantization_types"]
    assert "static" in quant_res["quantization_types"]
    assert quant_res["dynamic_model"] is not None
    assert quant_res["static_model"] is not None


# Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
def test_milestone_5():
    metrics = milestone_5(num_runs=2)
    assert isinstance(metrics, dict)
    assert "original" in metrics
    assert "dynamic" in metrics
    assert "static" in metrics
    for key in ("original", "dynamic", "static"):
        assert "size_mb" in metrics[key]
        assert "latency_ms" in metrics[key]
        assert "throughput_qps" in metrics[key]
        assert metrics[key]["size_mb"] > 0
        assert metrics[key]["latency_ms"] >= 0
    assert "size_reduction_pct" in metrics["dynamic"]
    assert "speedup" in metrics["dynamic"]
    assert "size_reduction_pct" in metrics["static"]
    assert "speedup" in metrics["static"]


# Milestone 6: Present a dashboard-like output of model efficiency metrics.
def test_milestone_6():
    dashboard = milestone_6()
    assert isinstance(dashboard, dict)
    assert "table" in dashboard
    assert "metrics" in dashboard
    assert "summary" in dashboard
    assert "recommendation" in dashboard
    assert len(dashboard["table"]) > 0
    assert "FP32" in dashboard["table"] or "Original" in dashboard["table"]
    assert "INT8" in dashboard["table"] or "Dynamic" in dashboard["table"]
