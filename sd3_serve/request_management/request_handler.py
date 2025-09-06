import sys
import os
import asyncio
import uuid
import torch
from datetime import datetime, timedelta

try:
    from scheduler import Scheduler
    from constants import RequestStatus
    from custom_serve.serving.request_processor import process_each_timestep
except ImportError:
    from .scheduler import Scheduler
    from .constants import RequestStatus
    from .request_processor import process_each_timestep, process_each_timestep_batched

sys.path.append("../")
from utils.logger import get_logger
from inference.sd3 import CFGDenoiser, SD3LatentFormat

PROFILE_GPU = os.getenv("PROFILE_GPU", "false").lower() == "true"


logger = get_logger(__name__)

class RequestPool:
    def __init__(self, inference_handler):
        self.raw_requests = {}
        self.requests = {}
        self.active_queue = asyncio.Queue()
        self.attn_queue = asyncio.Queue()
        self.decode_queue = asyncio.Queue()
        self.output_pool = asyncio.Queue()
        self.lock = asyncio.Lock()
        seed = torch.randint(0, 100000, (1,)).item()
        self.empty_latent = inference_handler.get_empty_latent(1, 1024, 1024, seed, device=inference_handler.device)
        self.neg_cond = inference_handler.fix_cond(inference_handler.get_cond(""))
        logger.info("Initialized RequestPool.")

    def add_request_to_pool(self, request):
        """Add a new request to the pool."""
        self.raw_requests[request["request_id"]] = request
        logger.info(f"Request added to pool: {request['request_id']} (Prompt: {request['prompt']})")

    def add_request_to_final_pool(self, request):
        """Add a new request to the pool."""
        self.requests[request["request_id"]] = request
        self.requests[request["request_id"]]["processing_time_start"] = datetime.now().isoformat()
        self.raw_requests.pop(request["request_id"])
        logger.info(f"Request added to final pool: {request['request_id']} (Prompt: {request['prompt']})")

    async def check_pending_timeouts(self, pending_timeout, current_time):
        """Check pending requests and mark those exceeding the timeout as failed."""
        for request_id, request in list(self.raw_requests.items()):
            request_time = datetime.fromisoformat(request["timestamp"])
            if current_time - request_time > timedelta(seconds=pending_timeout):
                for key in ["noise_scaled", "sigmas", "conditioning", "neg_cond", "old_denoised", "context_latent", "x_latent"]:
                    if key in request:
                        del request[key]
                del self.raw_requests[request_id]
                request["status"] = RequestStatus.FAILED
                logger.info(f"Request added to output queue: {request_id}")
                await self.add_to_output_pool(request)
       

    async def add_to_active_queue(self, request_id):
        """Add a request to the active queue."""
        async with self.lock:
            await self.active_queue.put(request_id)
        logger.debug(f"Request added to active queue: {request_id}")


    async def add_to_output_pool(self, request):
        """Add a completed request to the output pool."""
        async with self.lock:
            request['completed_timestamp'] = datetime.now().isoformat()
            await self.output_pool.put(request)
        logger.info(f"Request added to output pool: {request['request_id']} (Prompt: {request['prompt']})")


    async def get_all_active_requests(self):
        """Fetch all non-attention active requests."""
        requests = []
        async with self.lock:
            while not self.active_queue.empty():
                requests.append(await self.active_queue.get())
        logger.debug(f"Fetched {len(requests)} active requests.")
        return requests

    async def get_all_attn_requests(self):
        """Fetch all attention requests."""
        requests = []
        async with self.lock:
            while not self.attn_queue.empty():
                requests.append(await self.attn_queue.get())
        logger.debug(f"Fetched {len(requests)} attention requests.")
        return requests


class RequestHandler:
    def __init__(self, config=None, inference_handler=None):
        if config is None:
            config = {}
        sys_config = config["system"]
        self.request_pool = RequestPool(inference_handler)
        self.scheduler = Scheduler(batch_size=sys_config.get("batch_size", 1))
        self.cache_interval = sys_config.get("cache_interval", 5)
        self.max_requests = max(sys_config.get("batch_size", 1) * self.cache_interval + 1, 1)
        logger.info(f"Initialized RequestHandler with batch_size={self.scheduler.batch_size}, "
                    f"cache_interval={self.cache_interval}, max_requests={self.max_requests}")
        self.pending_timeout_check = sys_config.get("pending_timeout_check", 200)

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
        request = self.create_request(prompt, timesteps_left)
        self.request_pool.add_request_to_pool(request)

    def _prepare_first_timestep(self, inference_handler, request):
        prompt = request["prompt"]
        timesteps_left = request["timesteps_left"]
        empty_latent = self.request_pool.empty_latent
        neg_cond = self.request_pool.neg_cond
        elapsed_gpu_time = 0
        if PROFILE_GPU:
            noise_scaled, sigmas, conditioning, neg_cond, seed_num, elapsed_gpu_time = inference_handler.prepare_for_first_timestep(empty_latent, prompt, neg_cond, timesteps_left, seed_type="fixed")
        else:
            noise_scaled, sigmas, conditioning, neg_cond, seed_num = inference_handler.prepare_for_first_timestep(empty_latent, prompt, neg_cond, timesteps_left, seed_type="fixed")
        request["noise_scaled"] = noise_scaled
        request["sigmas"] = sigmas
        request["conditioning"] = conditioning
        request["neg_cond"] = neg_cond
        request["seed_num"] = seed_num
        request["old_denoised"] = None
        request["elapsed_gpu_time"] = elapsed_gpu_time
        return request
    
    def _prepare_and_add_to_final_pool(self, inference_handler, request_id):
        request = self.request_pool.raw_requests[request_id]
        request = self._prepare_first_timestep(inference_handler, request)
        self.request_pool.add_request_to_final_pool(request)

    async def prefill(self, inference_handler):
        await self.request_pool.check_pending_timeouts(self.pending_timeout_check, datetime.now())
        all_requests = list(self.request_pool.raw_requests.keys())
        final_requests_count = len(self.request_pool.requests)
        tasks = []
        for request_id in all_requests:
            if final_requests_count < self.max_requests:
                self._prepare_and_add_to_final_pool(inference_handler, request_id)
                final_requests_count += 1
            else:
                break
   
    def update_timesteps_left(self, request_id):
        if request_id in self.request_pool.requests:
            self.request_pool.requests[request_id]["timesteps_left"] -= 1
            self.request_pool.requests[request_id]["current_timestep"] += 1
            timesteps_left = self.request_pool.requests[request_id]["timesteps_left"]
            logger.info(f"Updated timesteps_left for request {request_id}: {timesteps_left}")

    def update_status(self, request):
        if request["timesteps_left"] == 0:
            request["status"] = RequestStatus.COMPLETED
            logger.info(f"Request {request['request_id']} marked as COMPLETED.")
        else:
            request["status"] = RequestStatus.IN_PROGRESS
            logger.debug(f"Request {request['request_id']} marked as IN_PROGRESS.")

    async def process_request(self, inference_handler, save_latents):
        logger.info("Starting request processing cycle...")
        inference_handler.denoiser = CFGDenoiser
        last_timeout_check = datetime.now()
        
        while True:
            if self.request_pool.raw_requests:
                await self.prefill(inference_handler)
            await self.scheduler.add_to_active_request_queue(self.request_pool, self.max_requests)
            await self.scheduler.shift_to_attn_queue(self.request_pool, self.max_requests)

            # Fetch all requests at once
            attn_requests = await self.request_pool.get_all_attn_requests()
            active_requests = await self.request_pool.get_all_active_requests()
            
            
            # Create two parallel tasks - one for attention requests, one for active requests
            tasks = []
            
            if attn_requests:
                # Process all attention requests in one task
                tasks.append(self._process_attention_batch(inference_handler, attn_requests, save_latents))
                
            if active_requests:
                # Process all non-attention requests in another task
                tasks.append(self._process_active_batch(inference_handler, active_requests, save_latents))
            
            # Execute both batches concurrently
            if tasks:
                await asyncio.gather(*tasks)
                # print("Each iteration time: ", time.time() - st)
            
            # while not self.request_pool.decode_queue.empty():
            #     request_id = await self.request_pool.decode_queue.get()
            #     asyncio.create_task(self._decode_request(inference_handler, request_id))

            # Update queue statuses after processing
            for request_id in attn_requests + active_requests:
                if request_id in self.request_pool.requests:
                    if self.request_pool.requests[request_id]["status"] != RequestStatus.COMPLETED:
                        await self.request_pool.add_to_active_queue(request_id)
                    else:
                        asyncio.create_task(self._decode_request(inference_handler, request_id))
                    # elif self.request_pool.requests[request_id]["status"] == RequestStatus.COMPLETED:
                    #     logger.debug(f"Request {request_id} completed. Moving to output pool.")
                    #     await self.request_pool.add_to_output_pool(self.request_pool.requests[request_id])
            
            # Check for timeouts periodically
            # if (datetime.now() - last_timeout_check).total_seconds() > self.pending_timeout_check:
            #     await self.request_pool.check_pending_timeouts(self.pending_timeout_check, datetime.now())
            #     last_timeout_check = datetime.now()
                
            await asyncio.sleep(0.001)      

    
    async def _decode_request(self, inference_handler, request_id):
        """
        Decodes the image for a request asynchronously without blocking the main process.
        """
        request = self.request_pool.requests[request_id]
        del self.request_pool.requests[request['request_id']]
        logger.debug(f"Request removed from active pool: {request['request_id']}")

        # Run decoding in a separate thread (prevents blocking)
        latent = SD3LatentFormat().process_out(request["noise_scaled"])
        image = inference_handler.vae_decode(latent)
        request["image"] = image  # Store decoded image

        # Clean up memory
        for key in ["noise_scaled", "sigmas", "conditioning", "neg_cond", "old_denoised", "context_latent", "x_latent"]:
            if key in request:
                del request[key]

        # Move request to output pool
        await self.request_pool.add_to_output_pool(request)


    async def _process_attention_batch(self, inference_handler, request_ids, save_latents):
        """Process a batch of attention requests asynchronously."""
        processed_requests = []
        # Process each attention request one by one
        for request_id in request_ids:
            # Use the existing sequential processing for attention requests
            request = self.request_pool.requests[request_id]
            
            # Run the potentially CPU-intensive process in a separate thread
            proc_request = await asyncio.to_thread(
                process_each_timestep,
                inference_handler,
                request_id,
                request,
                self.request_pool.neg_cond,
                cache_interval=self.cache_interval,
                compute_attention=True,
                save_latents=save_latents
            )
            self.request_pool.requests[request_id] = proc_request
            
            # Update request state
            proc_request["cache_interval"] = self.cache_interval
            proc_request["timesteps_left"] -= 1
            proc_request["current_timestep"] += 1
            
            # Handle completion
            # if request["timesteps_left"] == 0:
            #     await self.request_pool.decode_queue.put(request_id)
            
            self.update_status(proc_request)
            processed_requests.append(proc_request)
        
        return processed_requests

    async def _process_active_batch(self, inference_handler, request_ids, save_latents):
        """Process a batch of non-attention requests asynchronously."""
        # Use the existing batch processing for active requests
        requests = []
        for request_id in request_ids:
            requests.append(self.request_pool.requests[request_id])
        processed_requests = await asyncio.to_thread(
            process_each_timestep_batched,
            inference_handler,
            request_ids,
            requests,
            self.request_pool.neg_cond,
            cache_interval=self.cache_interval,
            compute_attention=False,
            save_latents=save_latents
        )
        
        # Update all processed requests
        for request in processed_requests:
            request_id = request["request_id"]
            request["cache_interval"] -= 1
            request["timesteps_left"] -= 1
            request["current_timestep"] += 1
            
            # Handle completion
            # if request["timesteps_left"] == 0:
            #     await self.request_pool.decode_queue.put(request_id)
            self.update_status(request)
            self.request_pool.requests[request_id] = request
        
        return processed_requests

    # def process_batch(self, inference_handler, request_ids, requires_attention):
    #     # if requires_attention:
    #     #     process_each_timestep_batched(inference_handler, request_ids, self.request_pool, compute_attention=True)
    #     # else:
    #     requests = process_each_timestep_batched(inference_handler, request_ids, self.request_pool, compute_attention=False)
    #     return requests
        


    # 
