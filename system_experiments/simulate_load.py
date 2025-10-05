import json
import random
import asyncio
import aiohttp
import os
import argparse
import time

# Config Defaults
DEFAULT_PROMPT_FILE = "/home/fast-dit-serving/partiprompts_generation/parti_prompts.json"
DEFAULT_COMPLETED_LOG_PATH = "/home/fast-dit-serving/system_experiments/completed_requests_rr_2_sec_100.json"
DEFAULT_SAVE_INTERVAL = 5


# Utility function to append new data to a JSON file safely
def append_to_json_file(new_requests, file_path):
    """
    Appends a list of completed requests to a JSON file.
    If the file doesn't exist or is empty, it initializes the structure.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {"completed_requests": []}
    else:
        existing_data = {"completed_requests": []}

    existing_data["completed_requests"].extend(new_requests)
    print(f"Dumping {len(new_requests)} requests to JSON")
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=2)


# Polling function to fetch completed requests periodically
async def poll_until_completed(get_url, completed_log_path, save_interval):
    """
    Continuously polls the server for completed requests and saves them periodically.
    """
    accumulated = {"completed_requests": []}
    last_flush_time = time.time()
    FLUSH_INTERVAL_SECONDS = 10  # Flush even if less than SAVE_INTERVAL

    try:
        while True:
            await asyncio.sleep(1)

            async with aiohttp.ClientSession() as session:
                async with session.get(get_url) as response:
                    if response.status == 200:
                        results = await response.json()
                        new_requests = results.get("completed_requests", [])

                        if new_requests:
                            for req in new_requests:
                                req_id = req.get("request_id")
                                existing_ids = {r.get("request_id") for r in accumulated["completed_requests"]}
                                if req_id not in existing_ids:
                                    accumulated["completed_requests"].append(req)
                                    print(f"Logged request {req_id}")

                            # Save if batch size reached
                            if len(accumulated["completed_requests"]) >= save_interval:
                                append_to_json_file(accumulated["completed_requests"], completed_log_path)
                                accumulated["completed_requests"].clear()

            # Timed flush even if batch not full
            if time.time() - last_flush_time > FLUSH_INTERVAL_SECONDS:
                if accumulated["completed_requests"]:
                    append_to_json_file(accumulated["completed_requests"], completed_log_path)
                    print(f"Timed flush: saved {len(accumulated['completed_requests'])} requests.")
                    accumulated["completed_requests"].clear()
                last_flush_time = time.time()

    except asyncio.CancelledError:
        print("Polling task cancelled.")
    except Exception as e:
        print(f"Error while polling: {e}")

    # Final save
    if accumulated["completed_requests"]:
        append_to_json_file(accumulated["completed_requests"], completed_log_path)
        print(f"Final flush: saved {len(accumulated['completed_requests'])} requests.")


# Function to send a single request
async def send_request(prompt_key, prompt_text, host):
    """
    Sends a prompt request to the server.
    """
    add_url = f"{host}/add_request"
    data = {
        "prompt": prompt_text,
        "timesteps_left": random.choice([50])
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(add_url, json=data) as response:
                if response.status == 200:
                    print(f"Submitted {prompt_key}")
                else:
                    print(f"Error submitting {prompt_key}: {response.status}")
    except Exception as e:
        print(f"Exception adding {prompt_key}: {e}")


# Load simulation functions
async def simulate_load(prompts, host, num_requests=10, delay_between=0.5):
    """
    Sends a fixed number of requests with a delay between them.
    """
    tasks = []
    for _ in range(num_requests):
        prompt_key, prompt_text = random.choice(prompts)
        tasks.append(send_request(prompt_key, prompt_text, host))
        await asyncio.sleep(delay_between)
    await asyncio.gather(*tasks)


async def simulate_burst_load(prompts, host, burst_size=20, burst_interval=5, total_duration=60):
    """
    Send bursts of requests: 'burst_size' back-to-back, then sleep 'burst_interval' seconds.
    """
    end_time = asyncio.get_event_loop().time() + total_duration
    burst_count = 0
    while asyncio.get_event_loop().time() < end_time:
        burst_count += 1
        print(f"Starting burst #{burst_count} ({burst_size} requests)")
        tasks = []
        for _ in range(burst_size):
            prompt_key, prompt_text = random.choice(prompts)
            tasks.append(send_request(prompt_key, prompt_text, host))
        await asyncio.gather(*tasks)
        print(f"Burst #{burst_count} complete. Sleeping for {burst_interval}s...\n")
        await asyncio.sleep(burst_interval)


async def simulate_constant_throughput(prompts, host, rate_per_sec=2, total_duration=30):
    """
    Sends requests at a constant rate (requests per second) for a total duration.
    """
    interval = 1.0 / rate_per_sec
    end_time = asyncio.get_event_loop().time() + total_duration
    while asyncio.get_event_loop().time() < end_time:
        prompt_key, prompt_text = random.choice(prompts)
        asyncio.create_task(send_request(prompt_key, prompt_text, host))
        await asyncio.sleep(interval)


# Main function with argument parsing
async def main():
    parser = argparse.ArgumentParser(description="Async request simulator with polling.")
    parser.add_argument("--prompt_file", default=DEFAULT_PROMPT_FILE, help="Path to the prompt JSON file.")
    parser.add_argument("--completed_log", default=DEFAULT_COMPLETED_LOG_PATH, help="Path to save completed requests log.")
    parser.add_argument("--save_interval", type=int, default=DEFAULT_SAVE_INTERVAL, help="Number of requests before saving batch.")
    parser.add_argument("--host", default="http://localhost:8000", help="Server host URL.")
    parser.add_argument("--rate", type=float, default=2, help="Requests per second for constant throughput.")
    parser.add_argument("--duration", type=int, default=100, help="Duration (seconds) for load simulation.")
    parser.add_argument("--mode", choices=["constant", "burst"], default="constant", help="Load simulation mode.")
    parser.add_argument("--burst_size", type=int, default=20)
    parser.add_argument("--burst_interval", type=float, default=5)

    args = parser.parse_args()

    # Load prompts from file
    prompts = list(json.load(open(args.prompt_file)).items())

    # Start background polling task
    get_url = f"{args.host}/get_output"
    poll_task = asyncio.create_task(poll_until_completed(get_url, args.completed_log, args.save_interval))

    # Run selected simulation mode
    if args.mode == "constant":
        await simulate_constant_throughput(prompts, args.host, args.rate, args.duration)
    else:
        await simulate_burst_load(prompts, args.host, args.burst_size, args.burst_interval, args.duration)

    # Allow time for all responses to finish
    await asyncio.sleep(1000)
    poll_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
