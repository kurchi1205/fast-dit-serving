import sys
import os
import json
import argparse
import asyncio
import aiohttp
from datetime import datetime
from PIL import Image
from collections import defaultdict

sys.path.insert(0, "../../")
from sd3_serve.utils.logger import get_logger
from sd3_serve.request_management.config_loader import ConfigLoader

logger = get_logger(__name__)

class CachingClient:
    def __init__(self, config_path="../sd3_serve/configs/config.yaml"):
        self.config = ConfigLoader(config_path).config
        self.server_url = self.config["server"]["url"]
        self.poll_interval = self.config["client"]["poll_interval"]

    async def start_background_process(self):
        url = f"{self.server_url}/start_background_process"
        params = {"model": ""}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Background process started: {result}")
                        return result
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to start background process: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error starting background process: {e}")
            return {"error": str(e)}

    async def change_caching_interval(self, interval: int):
        url = f"{self.server_url}/change_caching_interval"
        data = {"cache_interval": interval}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Caching interval changed: {result}")
                        return result
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to change caching interval: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Exception during change_caching_interval: {e}")
            return {"error": str(e)}

    async def add_request(self, prompt, timesteps_left):
        url = f"{self.server_url}/add_request"
        data = {"prompt": prompt, "timesteps_left": timesteps_left}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to add request. Server response: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            return {"error": str(e)}

    async def get_output(self):
        url = f"{self.server_url}/get_output"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("completed_requests", [])
                    else:
                        error_message = await response.text()
                        logger.error(f"Failed to fetch outputs. Server response: {error_message}")
                        return {"error": error_message}
        except Exception as e:
            logger.error(f"Error retrieving outputs: {e}")
            return {"error": str(e)}

    async def poll_for_outputs(self, path):
        while True:
            outputs = await self.get_output()
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            if isinstance(outputs, list) and outputs:
                for output in outputs:
                    start_time = datetime.fromisoformat(output["timestamp"])
                    end_time = datetime.fromisoformat(output["time_completed"])
                    time_taken = (end_time - start_time).total_seconds()
                    output["time_taken"] = time_taken
                    print(f"Completed Request: {json.dumps(output, indent=2)}")
                    image_path = output["image"]
                    img = Image.open(image_path)
                    img.save(path)
                break
            await asyncio.sleep(self.poll_interval)

    async def run(self, interval: int, prompt: str, timesteps_left: int, key: str):
        # logger.info("Starting background process...")
        # await self.start_background_process()

        # logger.info(f"Setting caching interval to {interval}")
        # await self.change_caching_interval(interval)

        logger.info("Submitting inference request...")
        await self.add_request(prompt, timesteps_left)

        logger.info("Polling for output...")
        path = f"/home/fast-dit-serving/assets/drawbench_sd3_custom/{key}_cache_{interval}.png"
        await self.poll_for_outputs(path)


    async def start_bg_process(self):
        logger.info("Starting background process...")
        await self.start_background_process()


async def process_prompts(client, prompts_dict, interval):
    for key, prompt in prompts_dict.items():
        print(f"Processing {key} (cache_{interval})")
        try:
            path = f"/home/fast-dit-serving/assets/drawbench_sd3_custom/{key}_cache_{interval}.png"
            if os.path.exists(path):
                print("continuing")
                continue
            await client.run(interval, prompt, timesteps_left=50, key=key)
        except Exception as e:
            logger.error(f"Error processing {key}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process prompts with caching intervals")
    parser.add_argument("--interval_list", type=int, nargs="+", default=[5], help="List of caching intervals")
    parser.add_argument("--prompt_path", type=str, default="drawbench_prompts.json", help="Path to prompts JSON file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    client = CachingClient()
    interval_list = args.interval_list
    prompt_path = args.prompt_path

    prompts = json.load(open(prompt_path))

    for interval in interval_list:
        asyncio.run(process_prompts(client, prompts, interval))