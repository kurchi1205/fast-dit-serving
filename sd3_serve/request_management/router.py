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
        self.lock = asyncio.Lock()
    
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
        # async with self.lock:
        #     await self.intake.put(request)
        
        async with self.lock:
            try:
                await self.intake.put(request)
                logger.info(f"Request added to queue: {request['request_id']} (Prompt: {request['prompt']})")
                return True
            except asyncio.QueueFull:
                return False



    def _assignable_slots(self, shard: GPUShard) -> int:
        """
        Keep each shard's (final + raw) <= max_requests so its own `prefill()`
        can move items from raw -> final without overload.
        """
        pool = shard["handler"].request_pool
        current_final = len(pool.requests)
        current_raw = len(pool.raw_requests)
        return max(0, shard["handler"].max_requests - (current_final + current_raw))
    

    def total_capacity_left(self) -> int:
        # How many requests can we still accept overall?
        shard_budget = sum(self._assignable_slots(s) for s in self.shards)
        intake_budget = self.max_queue - self.intake.qsize()
        # Be conservative: we only count space that can actually be serviced soon.
        return min(shard_budget + intake_budget, intake_budget)

    async def assign_requests_to_shards(self, poll_interval: float = 5):
        """
        Sequentially assign requests to GPUs based on their real-time assignable slots.
        Fill GPU-0 first until capacity is full, then move to GPU-1, and so on.
        """
        num_gpus = len(self.shards)
        self.current_gpu_idx = getattr(self, "current_gpu_idx", 0)

        while True:
            if self.intake.empty():
                await asyncio.sleep(0.01)
                continue

            made_progress = False
            num_pending = self.intake.qsize()
            for n in range(num_pending):
                assigned = False

                # Try to assign from current GPU onward
                tried = 0
                while not assigned:
                    tried += 1
                    gpu_idx = self.current_gpu_idx
                    shard = self.shards[gpu_idx]

                    # Compute assignable slots dynamically
                    slots = self._assignable_slots(shard)

                    if slots > 0:
                        # Assign request to this GPU
                        req = await self.intake.get()
                        req["device"] = shard["device"]
                        shard["handler"].request_pool.add_request_to_pool(req)
                        made_progress = True
                        assigned = True
                        break
                    else:
                        # Move to next GPU only when current is full
                        self.current_gpu_idx += 1
                        if self.current_gpu_idx == num_gpus:
                            self.current_gpu_idx = 0
                    if tried == num_gpus:
                        break

                # If all GPUs are full, reset to start and retry later
                if not assigned:
                    self.current_gpu_idx = 0
                    await asyncio.sleep(poll_interval)
                    break

            if not made_progress:
                await asyncio.sleep(poll_interval)


# class Router:
#     """
#     Central intake + distributor. Batches requests and distributes
#     efficiently across GPU shards.
#     """
#     def __init__(self, shards: List[GPUShard], max_queue_size: int = 1000):
#         assert shards, "Router needs at least one GPU shard"
#         self.shards = shards
#         self.intake: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
#         self.max_queue = max_queue_size
#         self.current_gpu_idx = 0
        
#         # Cache for GPU metrics (updated periodically)
#         self._gpu_slots_cache = [0] * len(shards)
#         self._cache_timestamp = 0
#         self._cache_ttl = 0.1  # Cache for 100ms
    
#     def create_request(self, prompt, timesteps_left):
#         request = {
#             "request_id": str(uuid.uuid4()),
#             "timestamp": datetime.now().isoformat(),
#             "status": RequestStatus.PENDING,
#             "current_timestep": 0,
#             "timesteps_left": timesteps_left,
#             "cache_interval": 0,
#             "prompt": prompt,
#             "cfg_scale": 5.0,
#             "context_latent": {},
#             "x_latent": {},
#             "elapsed_gpu_time": 0
#         }
#         logger.info(f"Created new request: {request['request_id']} (Prompt: {request['prompt']})")
#         return request
    
#     async def add_request(self, prompt, timesteps_left):
#         """Add a new request to the pool - non-blocking."""
#         request = self.create_request(prompt, timesteps_left)
        
#         try:
#             # Use put_nowait to avoid blocking
#             self.intake.put_nowait(request)
#             logger.info(f"Request added to queue: {request['request_id']} (Prompt: {request['prompt']})")
#             return True
#         except asyncio.QueueFull:
#             logger.warning("Intake queue is full, rejecting request")
#             return False

#     def _assignable_slots(self, shard: GPUShard) -> int:
#         """
#         Calculate how many requests this shard can accept.
#         Keep each shard's (final + raw) <= max_requests.
#         """
#         pool = shard["handler"].request_pool
#         current_final = len(pool.requests)
#         current_raw = len(pool.raw_requests)
#         max_requests = shard["handler"].max_requests
#         return max(0, max_requests - (current_final + current_raw))

#     def _update_gpu_slots_cache(self):
#         """Update cached GPU capacity information."""
#         current_time = asyncio.get_event_loop().time()
        
#         # Only update if cache expired
#         if current_time - self._cache_timestamp < self._cache_ttl:
#             return
        
#         for idx, shard in enumerate(self.shards):
#             self._gpu_slots_cache[idx] = self._assignable_slots(shard)
        
#         self._cache_timestamp = current_time

#     def total_capacity_left(self) -> int:
#         """Calculate total remaining capacity across all shards."""
#         self._update_gpu_slots_cache()
#         shard_budget = sum(self._gpu_slots_cache)
#         intake_budget = self.max_queue - self.intake.qsize()
#         return min(shard_budget + intake_budget, intake_budget)

#     async def assign_requests_to_shards(self, poll_interval: float = 0.1):
#         """
#         Efficiently batch-assign requests to GPUs.
#         Uses round-robin with dynamic load balancing.
#         """
#         num_gpus = len(self.shards)
        
#         while True:
#             # Wait for requests if queue is empty
#             if self.intake.empty():
#                 await asyncio.sleep(0.05)  # Longer sleep when idle
#                 continue
            
#             # Update GPU capacity cache
#             self._update_gpu_slots_cache()
            
#             # Collect batch of requests to assign
#             batch_size = min(self.intake.qsize(), sum(self._gpu_slots_cache))
#             if batch_size == 0:
#                 # All GPUs are full
#                 await asyncio.sleep(poll_interval)
#                 continue
            
#             # Batch extract requests from intake queue
#             requests_to_assign = []
#             for _ in range(batch_size):
#                 try:
#                     req = self.intake.get_nowait()
#                     requests_to_assign.append(req)
#                 except asyncio.QueueEmpty:
#                     break
            
#             if not requests_to_assign:
#                 await asyncio.sleep(0.01)
#                 continue
            
#             # Distribute requests using round-robin with capacity awareness
#             assigned_count = 0
#             attempts = 0
#             max_attempts = num_gpus * 2
            
#             for req in requests_to_assign:
#                 assigned = False
                
#                 while not assigned and attempts < max_attempts:
#                     gpu_idx = self.current_gpu_idx
#                     shard = self.shards[gpu_idx]
                    
#                     # Check if this GPU has capacity (from cache)
#                     if self._gpu_slots_cache[gpu_idx] > 0:
#                         # Assign request
#                         shard["handler"].request_pool.add_request_to_pool(req)
#                         self._gpu_slots_cache[gpu_idx] -= 1  # Update cache
#                         assigned_count += 1
#                         assigned = True
                        
#                         # Move to next GPU for round-robin
#                         self.current_gpu_idx = (self.current_gpu_idx + 1) % num_gpus
#                     else:
#                         # Try next GPU
#                         self.current_gpu_idx = (self.current_gpu_idx + 1) % num_gpus
#                         attempts += 1
                
#                 if not assigned:
#                     # Put back in queue if couldn't assign
#                     try:
#                         self.intake.put_nowait(req)
#                     except asyncio.QueueFull:
#                         logger.error(f"Failed to reassign request {req['request_id']}")
#                     break
            
#             logger.debug(f"Assigned {assigned_count}/{len(requests_to_assign)} requests")
            
#             # Brief yield to event loop
#             await asyncio.sleep(0)