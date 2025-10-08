import asyncio
import time
from datetime import datetime
import aiohttp
import numpy as np


class BenchmarkClient:
    def __init__(self):
        self.server_url = "http://localhost:8000"
        self.poll_interval = 0.1  # Shorter polling interval for benchmarking
        self.empty_pool_timeout = 30  # Timeout after 30 seconds of empty pool
        self.latencies = []

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

    async def poll_with_timeout(self):
        start_empty_time = None
        start_time = time.time()

        while True:
            outputs = await self.get_output()
            
            if outputs:
                output = outputs[0]
                start_time = datetime.fromisoformat(output["timestamp"])
                end_time = datetime.fromisoformat(output["time_completed"])
                time_taken = (end_time - start_time).total_seconds()
                return output, time_taken
            
            if not outputs:
                if start_empty_time is None:
                    start_empty_time = time.time()
                elif time.time() - start_empty_time > self.empty_pool_timeout:
                    return None, None
            else:
                start_empty_time = None
                
            await asyncio.sleep(self.poll_interval)

    async def single_iteration(self, prompt, timesteps):
        try:
            await self.add_request(prompt, timesteps)
            result, latency = await self.poll_with_timeout()
            
            if result and latency:
                return {
                    "latency": latency,
                    "result": result
                }
            return None

        except Exception as e:
            return None

    def calculate_statistics(self):
        if not self.latencies:
            return None
            
        latencies_array = np.array(self.latencies)
        return {
            "mean": np.mean(latencies_array),
            "median": np.median(latencies_array),
            "std": np.std(latencies_array),
            "min": np.min(latencies_array),
            "max": np.max(latencies_array),
            "p95": np.percentile(latencies_array, 95),
            "p99": np.percentile(latencies_array, 99),
        }

    async def run_benchmark(self, iterations=10, prompt="Generate an image of a cat", timesteps=30):
        try:
            # Start the background process
            # await self.start_background_process()
            
            
            for i in range(iterations):
                
                # Wait for 1 second between iterations to avoid overwhelming the server
                if i > 0:
                    await asyncio.sleep(1)
                
                result = await self.single_iteration(prompt, timesteps)
                
                if result:
                    self.latencies.append(result["latency"])
                    

            # Calculate and return statistics
            stats = self.calculate_statistics()
            if stats:
                return {
                    "statistics": stats,
                    "raw_latencies": self.latencies,
                    "successful_iterations": len(self.latencies),
                    "total_iterations": iterations
                }
            return None

        except Exception as e:
            return None

async def main():
    # Configure these parameters as needed
    ITERATIONS = 20
    PROMPT = "Generate an image of a cat"
    TIMESTEPS = 50

    client = BenchmarkClient()
    result = await client.run_benchmark(
        iterations=ITERATIONS,
        prompt=PROMPT,
        timesteps=TIMESTEPS
    )
    
    if result:
        stats = result["statistics"]
        print("\nBenchmark Summary:")
        print(f"Successful iterations: {result['successful_iterations']}/{result['total_iterations']}")
        print("\nLatency Statistics (seconds):")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Std Dev: {stats['std']:.2f}")
        print(f"Min: {stats['min']:.2f}")
        print(f"Max: {stats['max']:.2f}")
        print(f"95th percentile: {stats['p95']:.2f}")
        print(f"99th percentile: {stats['p99']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())