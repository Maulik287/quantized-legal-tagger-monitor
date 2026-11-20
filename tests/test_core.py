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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Apply dynamic and static post-training quantization to the trained model.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Present a dashboard-like output of model efficiency metrics.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
