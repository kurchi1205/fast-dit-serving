import asyncio
import time
from datetime import datetime
import aiohttp
import numpy as np


class BenchmarkClient:
    def __init__(self, config_path="../../configs/config.yaml"):
        self.server_url = "http://localhost:8000"
        self.poll_interval = 0.1  # Shorter polling interval for benchmarking
        self.empty_pool_timeout = 30  # Timeout after 30 seconds of empty pool
        self.batch_latencies = []  # Store latencies for each batch
        self.individual_latencies = []  # Store latencies for individual requests

    async def add_request(self, prompt, timesteps_left):
        url = f"{self.server_url}/add_request"
        data = {"prompt": prompt, "timesteps_left": timesteps_left}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_message = await response.text()
                    raise Exception(f"Failed to add request: {error_message}")

    async def get_output(self):
        url = f"{self.server_url}/get_output"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("completed_requests", [])
                else:
                    error_message = await response.text()
                    raise Exception(f"Failed to get output: {error_message}")

    async def start_background_process(self):
        url = f"{self.server_url}/start_background_process"
        params = {"model": ""}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_message = await response.text()
                    raise Exception(f"Failed to start background process: {error_message}")

    async def poll_batch_with_timeout(self, batch_size):
        start_empty_time = None
        completed_requests = []
        batch_start_time = time.time()

        while len(completed_requests) < batch_size:
            outputs = await self.get_output()
            
            if outputs:
                for output in outputs:
                    if output not in completed_requests:
                        start_time = datetime.fromisoformat(output["timestamp"])
                        end_time = datetime.fromisoformat(output["time_completed"])
                        time_taken = (end_time - start_time).total_seconds()
                        completed_requests.append({
                            "result": output,
                            "latency": time_taken
                        })
                start_empty_time = None
            else:
                if start_empty_time is None:
                    start_empty_time = time.time()
                elif time.time() - start_empty_time > self.empty_pool_timeout:
                    return None

            if len(completed_requests) < batch_size:
                await asyncio.sleep(self.poll_interval)

        batch_end_time = time.time()
        batch_latency = batch_end_time - batch_start_time
        
        return completed_requests, batch_latency

    async def single_batch_iteration(self, prompts, timesteps):
        try:
            # Submit all requests in the batch
            for prompt in prompts:
                await self.add_request(prompt, timesteps)
            
            # Wait for all results
            results = await self.poll_batch_with_timeout(len(prompts))
            
            if results:
                completed_requests, batch_latency = results
                return {
                    "batch_latency": batch_latency,
                    "individual_results": completed_requests
                }
            return None

        except Exception as e:
            return None

    def calculate_statistics(self, latencies):
        if not latencies:
            return None
            
        latencies_array = np.array(latencies)
        return {
            "mean": np.mean(latencies_array),
            "median": np.median(latencies_array),
            "std": np.std(latencies_array),
            "min": np.min(latencies_array),
            "max": np.max(latencies_array),
            "p95": np.percentile(latencies_array, 95),
            "p99": np.percentile(latencies_array, 99),
        }

    async def run_benchmark(self, iterations=10, batch_size=2, base_prompt="Generate an image of", timesteps=30):
        try:
            # Start the background process
            # await self.start_background_process()
            
            
            for i in range(iterations):
                
                # Wait between iterations to avoid overwhelming the server
                if i > 0:
                    await asyncio.sleep(2)  # Increased cool-down period for batches
                
                # Generate unique prompts for this batch
                prompts = [f"{base_prompt} {j+1}" for j in range(batch_size)]
                
                result = await self.single_batch_iteration(prompts, timesteps)
                
                if result:
                    self.batch_latencies.append(result["batch_latency"])
                    for individual_result in result["individual_results"]:
                        self.individual_latencies.append(individual_result["latency"])
                

            # Calculate and return statistics
            batch_stats = self.calculate_statistics(self.batch_latencies)
            individual_stats = self.calculate_statistics(self.individual_latencies)
            
            if batch_stats and individual_stats:
                return {
                    "batch_statistics": batch_stats,
                    "individual_statistics": individual_stats,
                    "raw_batch_latencies": self.batch_latencies,
                    "raw_individual_latencies": self.individual_latencies,
                    "successful_iterations": len(self.batch_latencies),
                    "total_iterations": iterations
                }
            return None

        except Exception as e:
            return None

async def main():
    # Configure these parameters as needed
    ITERATIONS = 1
    BATCH_SIZE = 24
    BASE_PROMPT = "Generate an image of a cat"
    TIMESTEPS = 50

    client = BenchmarkClient()
    result = await client.run_benchmark(
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        base_prompt=BASE_PROMPT,
        timesteps=TIMESTEPS
    )
    
    if result:
        print("\nBenchmark Summary:")
        print(f"Successful iterations: {result['successful_iterations']}/{result['total_iterations']}")
        
        print("\nBatch Latency Statistics (seconds):")
        batch_stats = result["batch_statistics"]
        print(f"Mean: {batch_stats['mean']:.2f}")
        print(f"Median: {batch_stats['median']:.2f}")
        print(f"Std Dev: {batch_stats['std']:.2f}")
        print(f"Min: {batch_stats['min']:.2f}")
        print(f"Max: {batch_stats['max']:.2f}")
        print(f"95th percentile: {batch_stats['p95']:.2f}")
        print(f"99th percentile: {batch_stats['p99']:.2f}")
        
        print("\nIndividual Request Latency Statistics (seconds):")
        ind_stats = result["individual_statistics"]
        print(f"Mean: {ind_stats['mean']:.2f}")
        print(f"Median: {ind_stats['median']:.2f}")
        print(f"Std Dev: {ind_stats['std']:.2f}")
        print(f"Min: {ind_stats['min']:.2f}")
        print(f"Max: {ind_stats['max']:.2f}")
        print(f"95th percentile: {ind_stats['p95']:.2f}")
        print(f"99th percentile: {ind_stats['p99']:.2f}")
        

if __name__ == "__main__":
    asyncio.run(main())