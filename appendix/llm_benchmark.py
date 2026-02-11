import time
import logging
import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

from openai import OpenAI

# Standardized logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("llm_benchmark")
RESULTS_CSV = os.environ.get("LLM_BENCHMARK_CSV", "/tmp/benchmark_results.csv")


def benchmark_endpoint(url, model, prompt, max_tokens=512, api_key="ollama"):
    """Run a simple streaming benchmark against an OpenAI-compatible API.

    Returns a dict with timing metrics and the full generated text.
    """
    client = OpenAI(base_url=url, api_key=api_key)  # API key ignored by many local providers

    logger.info("Starting request to %s (model=%s)", url, model)
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        )
    except Exception as e:
        logger.exception("Request failed: %s", e)
        raise

    first_token_time = None
    tokens = 0
    response_text = ""
    chunk_count = 0

    # Collect streamed deltas and compute simple metrics
    for chunk in response:
        chunk_count += 1
        # Debug: log the chunk structure
        logger.debug("Chunk %d: %s", chunk_count, chunk)
        
        content = None
        try:
            # Try to access content from chunk.choices[0].delta
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                # Some models (like reasoning models) use delta.reasoning instead of delta.content
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                elif hasattr(delta, "reasoning") and delta.reasoning:
                    content = delta.reasoning
        except (IndexError, AttributeError) as e:
            logger.debug("Error accessing chunk.choices[0].delta: %s", e)

        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            response_text += content
            # Rough token estimation: count words + punctuation
            tokens += len(content.split())
        else:
            logger.debug("No content in chunk %d", chunk_count)

    end_time = time.perf_counter()

    total_duration = end_time - start_time
    generation_time = (end_time - first_token_time) if first_token_time is not None else 0.0
    ttft = (first_token_time - start_time) if first_token_time is not None else total_duration
    tps = tokens / generation_time if generation_time > 0 else 0.0

    results = {
        "ttft": ttft,
        "tps": tps,
        "total_tokens": tokens,
        "duration": total_duration,
        "response_text": response_text,
    }

    logger.info(
        "Finished request: ttft=%.4fs tps=%.2f total_tokens=%d duration=%.4fs",
        results["ttft"],
        results["tps"],
        results["total_tokens"],
        results["duration"],
    )

    return results


def save_results_csv(ep, results, csv_path=RESULTS_CSV):
    """Append a result row to CSV file.

    Fields: timestamp, name, url, model, ttft, tps, total_tokens, duration
    """
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "name", "url", "model", "ttft", "tps", "total_tokens", "duration"])
        writer.writerow([
            datetime.now().isoformat(),
            ep.get("name", ""),
            ep.get("url", ""),
            ep.get("model", ""),
            f"{results['ttft']:.6f}",
            f"{results['tps']:.6f}",
            results["total_tokens"],
            f"{results['duration']:.6f}",
        ])


def report_from_csv(csv_path=RESULTS_CSV):
    """Read CSV and print a comparison table aggregated by name+model."""
    if not os.path.exists(csv_path):
        print(f"No results file found at {csv_path}")
        return

    # Read all rows first so we can print raw results then aggregate
    rows_list = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_list.append(row)

    # Print raw results table
    print("\nRaw Results:\n")
    print(f"{'Timestamp':30}{'Name':20}{'Model':30}{'TTFT':>10}{'TPS':>10}{'Tokens':>8}{'Duration':>10}")
    print("-" * 115)
    for row in rows_list:
        try:
            ttft = float(row.get("ttft") or 0)
            tps = float(row.get("tps") or 0)
            tokens = int(row.get("total_tokens") or 0)
            dur = float(row.get("duration") or 0)
        except ValueError:
            continue
        ts = row.get("timestamp", "")
        name = row.get("name", "")
        model = row.get("model", "")
        print(f"{ts:30.30}{name:20.20}{model:30.30}{ttft:10.4f}{tps:10.2f}{tokens:8d}{dur:10.4f}")

    # Aggregate by (name, model)
    data = defaultdict(list)
    for row in rows_list:
        key = (row.get("name", ""), row.get("model", ""))
        try:
            ttft = float(row.get("ttft") or 0)
            tps = float(row.get("tps") or 0)
            dur = float(row.get("duration") or 0)
        except ValueError:
            continue
        data[key].append((ttft, tps, dur))

    # Print header
    print("\nBenchmark Summary:\n")
    print(f"{'Name':30}{'Model':30}{'Runs':>6}{'Avg TPS':>10}{'Avg TTFT':>12}{'Avg Dur':>12}")
    print("-" * 100)
    for (name, model), rows in sorted(data.items()):
        runs = len(rows)
        avg_tps = sum(r[1] for r in rows) / runs if runs else 0
        avg_ttft = sum(r[0] for r in rows) / runs if runs else 0
        avg_dur = sum(r[2] for r in rows) / runs if runs else 0
        print(f"{name:30}{model:30}{runs:6d}{avg_tps:10.2f}{avg_ttft:12.4f}{avg_dur:12.4f}")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LLM benchmark tool")
    parser.add_argument("--report", action="store_true", help="Show aggregated report from results CSV and exit")
    parser.add_argument("--csv", default=RESULTS_CSV, help="Path to results CSV file")
    parser.add_argument("--model", help="Override model for all endpoints (optional)")
    parser.add_argument("--url", help="Override URL for all endpoints (optional)")
    args = parser.parse_args()

    # If report requested, print report and exit
    if args.report:
        report_from_csv(args.csv)
        raise SystemExit(0)

    # Example Usage
    endpoints = [
        {"name": "Ollama", "url": "http://localhost:11434/v1", "model": "qwen3:8b"},
        {"name": "TensorRT-LLM", "url": "http://localhost:8001/v1", "model": "nvidia/Qwen3-8B-NVFP4"},
        {"name": "vLLM", "url": "http://localhost:8002/v1", "model": "/models/Qwen3-8B-NVFP4"},
    ]

    # If a model or URL override was provided on the CLI, apply them to all endpoints
    if args.model:
        for e in endpoints:
            e["model"] = args.model
    if args.url:
        for e in endpoints:
            e["url"] = args.url

    prompt = "Explain the theory of relativity in detail."

    def choose_endpoints_menu(endpoints):
        """Present a simple numbered menu and return the selected endpoints list.

        Accepts a single number (e.g. `2`), a comma-separated list (`1,3`),
        or `q` to quit.
        """
        print("Select an endpoint to benchmark:")
        for i, e in enumerate(endpoints, start=1):
            print(f"  {i}) {e['name']} - {e['url']} (model={e['model']})")
        print("  q) Quit")

        while True:
            choice = input("Enter choice (number, comma-list, 'q' to quit): ").strip().lower()
            if choice == 'q':
                raise SystemExit(0)

            parts = [p.strip() for p in choice.split(',') if p.strip()]
            indices = []
            valid = True
            for p in parts:
                if not p.isdigit():
                    valid = False
                    break
                idx = int(p)
                if idx < 1 or idx > len(endpoints):
                    valid = False
                    break
                indices.append(idx - 1)

            if valid and indices:
                # keep order & remove duplicates
                seen = set()
                selected = []
                for idx in indices:
                    if idx not in seen:
                        selected.append(endpoints[idx])
                        seen.add(idx)
                return selected

            print("Invalid selection; try again.")

    selected_endpoints = choose_endpoints_menu(endpoints)

    for ep in selected_endpoints:
        logger.info("Next endpoint: %s (%s)", ep["name"], ep["url"]) 
        logger.info("Connecting to %s", ep["name"])
        logger.info("Benchmarking %s", ep["name"]) 
        try:
            res = benchmark_endpoint(ep["url"], ep["model"], prompt)
        except Exception:
            logger.error("Benchmark failed for %s", ep["name"])
            continue

        # Save results to CSV
        try:
            save_results_csv(ep, res, csv_path=args.csv)
        except Exception:
            logger.exception("Failed to save results to CSV %s", args.csv)

        logger.info("Results for %s: TPS=%.2f TTFT=%.4fs Duration=%.4fs Tokens=%d",
                    ep["name"], res["tps"], res["ttft"], res["duration"], res["total_tokens"])

        # Output the generated response (may be large)
        print("\n--- Response from %s ---" % ep["name"]) 
        print(res["response_text"])
        print("--- End Response ---\n")