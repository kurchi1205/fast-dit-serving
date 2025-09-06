import sys
import asyncio
import uuid
from datetime import datetime
from typing import List

try:
    from constants import RequestStatus

except ImportError:
    from .constants import RequestStatus

sys.path.append("../")
from utils.logger import get_logger
logger = get_logger(__name__)

class GPUShard:
    idx: int
    handler: "RequestHandler"


class Router:
    """
    Central intake + distributor. It only enqueues RAW requests
    into each GPU's RequestPool.raw_requests. The GPU's own loop
    will 'prefill' and prepare them on-device.
    """
    def __init__(self, shards: List[GPUShard], logger=None):
        assert shards, "Router needs at least one GPU shard"
        self.shards = shards
        self.intake: asyncio.Queue = asyncio.Queue()

    
    def create_request(self, prompt, timesteps_left):
        request = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": RequestStatus.PENDING,
            "current_timestep": 0,
            "timesteps_left": timesteps_left,
            "cache_interval": 0,  # Default cache interval
            "prompt": prompt,
            "cfg_scale": 5.0,
            "context_latent": {},
            "x_latent": {},
            "elapsed_gpu_time": 0
        }
        logger.info(f"Created new request: {request['request_id']} (Prompt: {request['prompt']})")
        return request
    
    
    async def add_request(self, prompt, timesteps_left):
        """Add a new request to the pool."""
        request = self.create_request(prompt, timesteps_left)
        self.intake.put(request)
        logger.info(f"Request added to queue: {request['request_id']} (Prompt: {request['prompt']})")

    
    def _assignable_slots(self, shard: GPUShard) -> int:
        """
        Keep each shard's (final + raw) <= max_requests so its own `prefill()`
        can move items from raw -> final without overload.
        """
        pool = shard.handler.request_pool
        current_final = len(pool.requests)
        current_raw = len(pool.raw_requests)
        return max(0, shard.handler.max_requests - (current_final + current_raw))
    

    async def assign_requests_to_shards(self, poll_interval: float = 5):
        """
        Drain the central intake and distribute RAW requests to shards'
        raw pools, similar in spirit to your `prefill` logic.
        Run this forever as a background task.
        """
        while True:
            made_progress = False

            if self.intake.empty():
                await asyncio.sleep(poll_interval)
                continue

            # Fill each shard up to capacity
            for shard in self.shards:
                slots = self._assignable_slots(shard)
                if slots <= 0:
                    continue

                while slots > 0 and not self.intake.empty():
                    req = await self.intake.get()
                    shard.handler.request_pool.add_request_to_pool(req)
                    slots -= 1
                    made_progress = True

            if not made_progress:
                # No shard had room; sleep for a bit
                await asyncio.sleep(poll_interval)
