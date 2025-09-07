"""
Simple batch test client for SD3 server (with background process)
Usage:
  python run_batched_example.py prompts.json [-s http://localhost:8000] [-t 600] [-o results.json] [--model MODEL]
"""

import argparse
import asyncio
import aiohttp
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


# ----------------------------
# Helpers
# ----------------------------

def load_prompts(file_path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSON (list[str], list[{'prompt','timesteps'}], or {'prompts': [...]})"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {file_path}: {e}")
        sys.exit(1)

    raw = data if isinstance(data, list) else data.get("prompts")
    if not isinstance(raw, list):
        print("JSON must be a list of prompts or have a 'prompts' key")
        sys.exit(1)

    prompts: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            prompts.append({"prompt": item, "timesteps": 30})
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append({"prompt": item["prompt"], "timesteps": item.get("timesteps", 30)})
    print(f"Loaded {len(prompts)} prompts")
    return prompts


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to JSON file (make fields serializable)"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if k == "image" and v is not None:
                clean[k] = str(v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        serializable.append(clean)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_requests": len(results),
                "results": serializable,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results saved to {output_file}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("\nNo results to summarize")
        return

    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = len(results) - completed
    total_gpu = sum(float(r.get("elapsed_gpu_time", 0)) for r in results)
    avg_gpu = (total_gpu / len(results)) if results else 0.0

    print("\n" + "=" * 60)
    print("BATCH TEST SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {len(results)}")
    print(f"Completed:   {completed}")
    print(f"Failed:      {failed}")

    if completed:
        print("\nGenerated Images:")
        for i, r in enumerate(results, start=1):
            if r.get("status") == "completed" and r.get("image"):
                p = r.get("prompt", "Unknown")
                pprev = (p[:50] + "...") if len(p) > 50 else p
                print(f"  {i}. {pprev}")
                print(f"     Image: {r['image']}")
                print(f"     GPU Time: {r.get('elapsed_gpu_time', 'N/A')}s")
    print("=" * 60 + "\n")


# ----------------------------
# Network ops
# ----------------------------

async def test_connection(session: aiohttp.ClientSession, server_url: str) -> bool:
    try:
        async with session.get(f"{server_url}/get_output", timeout=5) as resp:
            print(f"Server status: {resp.status}")
            return True
    except Exception as e:
        print(f"Cannot reach server at {server_url}: {e}")
        return False


async def start_background_process(session: aiohttp.ClientSession, server_url: str, model: str = "") -> bool:
    """Kick off the server's background worker."""
    try:
        params = {"model": model} if model is not None else {}
        async with session.post(f"{server_url}/start_background_process", params=params) as resp:
            if resp.status == 200:
                print("Background process started")
                return True
            else:
                txt = await resp.text()
                print(f"Background process response {resp.status}: {txt}")
                return False
    except Exception as e:
        print(f"Error starting background process: {e}")
        return False


async def submit_request(session: aiohttp.ClientSession, server_url: str, prompt: str, timesteps: int) -> Dict[str, Any]:
    data = {"prompt": prompt, "timesteps_left": timesteps}
    try:
        async with session.post(f"{server_url}/add_request", json=data) as resp:
            if resp.status == 200:
                j = await resp.json()
                return {"success": True, "data": j}
            else:
                return {"success": False, "error": await resp.text()}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def submit_all(session: aiohttp.ClientSession, server_url: str, prompts: List[Dict[str, Any]]) -> int:
    tasks = [submit_request(session, server_url, p["prompt"], p["timesteps"]) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"Submitted {ok}/{len(prompts)} requests successfully")
    return ok


async def poll_results(session: aiohttp.ClientSession, server_url: str, expected: int, timeout: int) -> List[Dict[str, Any]]:
    print(f"Waiting for {expected} results (timeout {timeout}s)...")
    start = time.time()
    done: List[Dict[str, Any]] = []

    while time.time() - start < timeout:
        try:
            async with session.get(f"{server_url}/get_output") as resp:
                if resp.status == 200:
                    j = await resp.json()
                    new_items = j.get("completed_requests", [])
                    if new_items:
                        done.extend(new_items)
                        print(f"Received {len(new_items)} new -> {len(done)}/{expected}")
                        if len(done) >= expected:
                            return done
        except Exception:
            pass
        await asyncio.sleep(5)

    print(f"Timeout. Received {len(done)}/{expected}")
    return done


# ----------------------------
# Main
# ----------------------------

async def amain() -> None:
    parser = argparse.ArgumentParser(description="Simple batch test client for SD3 server (with background process)")
    parser.add_argument("prompts_file", help="JSON file with prompts")
    parser.add_argument("-s", "--server-url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("-t", "--timeout", type=int, default=600, help="Timeout for all results (sec)")
    parser.add_argument("-o", "--output", default="batch_results.json", help="Where to save results")
    parser.add_argument("--model", default="", help="Model name to pass to /start_background_process")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"Starting with {len(prompts)} prompts")
    print(f"Server: {args.server_url}")
    print(f"Timeout: {args.timeout}s")
    print("-" * 50)

    async with aiohttp.ClientSession() as session:
        if not await test_connection(session, args.server_url):
            print("Server connection failed. Is it running?")
            sys.exit(1)

        # # Start background worker
        # if not await start_background_process(session, args.server_url, args.model):
        #     print("Proceeding even though background process did not confirm start...")

        submitted = await submit_all(session, args.server_url, prompts)
        if submitted == 0:
            print("No requests submitted successfully")
            sys.exit(1)

        results = await poll_results(session, args.server_url, submitted, args.timeout)
        if results:
            save_results(results, args.output)
            print_summary(results)
        else:
            print("No completed results received")
            sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
