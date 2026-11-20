"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantized-legal-tagger-monitor",
        description="Legal aid workers use this to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    
    # Milestone 1: Prepare data
    p_data = sub.add_parser("prepare-data", help="Prepare subset of LEDGAR dataset")
    p_data.add_argument("--samples", type=int, default=15, help="Number of samples to prepare")
    
    # Milestone 2: Train / Fine-tune
    p_train = sub.add_parser("train", help="Fine-tune GPT-2 model on legal clauses")
    p_train.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p_train.add_argument("--steps", type=int, default=5, help="Max training steps")
    
    # Milestone 3: Evaluate
    sub.add_parser("evaluate", help="Evaluate legal clause tagging accuracy")
    
    # Milestone 4: Quantize
    sub.add_parser("quantize", help="Apply dynamic and static post-training quantization")
    
    # Milestone 5: Benchmark
    p_bench = sub.add_parser("benchmark", help="Compare model sizes and simulate inference speeds")
    p_bench.add_argument("--runs", type=int, default=5, help="Number of benchmark iterations")
    
    # Milestone 6: Dashboard
    sub.add_parser("dashboard", help="Display comparative efficiency dashboard")
    
    # Run all
    sub.add_parser("run-all", help="Run complete end-to-end legal tagger pipeline")
    
    args = parser.parse_args(argv)
    
    if args.command == "hello":
        print("Quantized Legal Tagger Monitor is ready. Open README.md for your milestones.")
    elif args.command == "prepare-data":
        res = milestone_1(num_samples=args.samples)
        print(f"Prepared {res['count']} LEDGAR legal clause samples across {len(res['tags'])} tags.")
    elif args.command == "train":
        print("Fine-tuning GPT-2 legal tagger...")
        res = milestone_2(epochs=args.epochs, max_steps=args.steps)
        print(f"Training completed. Final loss: {res['loss']}, Steps: {res['steps']}.")
    elif args.command == "evaluate":
        print("Evaluating legal clause tagging accuracy...")
        res = milestone_3()
        print(f"Accuracy: {res['accuracy'] * 100:.1f}% ({res['correct']}/{res['total']} correct)")
    elif args.command == "quantize":
        print("Applying post-training dynamic and static quantization...")
        res = milestone_4()
        print(f"Quantization successful. Types: {', '.join(res['quantization_types'])}")
    elif args.command == "benchmark":
        print("Benchmarking FP32 vs INT8 Dynamic vs INT8 Static...")
        metrics = milestone_5(num_runs=args.runs)
        print(f"Original size: {metrics['original']['size_mb']:.3f} MB, Latency: {metrics['original']['latency_ms']:.2f} ms")
        print(f"Dynamic INT8 size: {metrics['dynamic']['size_mb']:.3f} MB (-{metrics['dynamic']['size_reduction_pct']}%), Speedup: {metrics['dynamic']['speedup']}x")
        print(f"Static INT8 size: {metrics['static']['size_mb']:.3f} MB (-{metrics['static']['size_reduction_pct']}%), Speedup: {metrics['static']['speedup']}x")
    elif args.command == "dashboard":
        res = milestone_6()
        print("\n" + res["table"] + "\n")
        print(res["summary"] + "\n")
        print(res["recommendation"])
    elif args.command == "run-all":
        print("Running end-to-end Quantized Legal Tagger Monitor pipeline...")
        d1 = milestone_1()
        print(f"1. Prepared {d1['count']} samples.")
        d2 = milestone_2()
        print(f"2. Fine-tuned model (loss: {d2['loss']}).")
        d3 = milestone_3(model=d2["model"], tokenizer=d2["tokenizer"])
        print(f"3. Evaluated accuracy: {d3['accuracy'] * 100:.1f}%.")
        d4 = milestone_4(model=d2["model"])
        print("4. Quantized dynamically and statically.")
        d5 = milestone_5(models_dict=d4, tokenizer=d2["tokenizer"])
        print("5. Benchmarked model efficiency.")
        d6 = milestone_6(metrics=d5)
        print("\n" + d6["table"] + "\n")
        print(d6["summary"])


if __name__ == "__main__":
    main()
