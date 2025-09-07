#!/usr/bin/env python3
"""
Ultra-simple test client for SD3 server
Usage: python simple_test.py "your prompt here"
"""
import argparse
import asyncio
import aiohttp
import sys
import time
import json


async def test_server(prompt, server_url="http://localhost:8000", timesteps=30):
    """Submit request and wait for result"""
    
    print(f"🚀 Running server with prompt: '{prompt}'")
    print(f"Server: {server_url}")
    print("-" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            # # 1. Start background process
            # print("1. Starting background process...")
            # async with session.post(f"{server_url}/start_background_process", params={"model": ""}) as resp:
            #     if resp.status == 200:
            #         print("Background process started")
            #     else:
            #         print(f"Background process response: {resp.status}")
            
            # 2. Submit request
            print("2. Submitting inference request...")
            data = {"prompt": prompt, "timesteps_left": timesteps}
            async with session.post(f"{server_url}/add_request", json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"Request submitted: {result}")
                else:
                    error = await resp.text()
                    print(f"Request failed: {error}")
                    return False
            
            # 3. Poll for results
            print("3. Waiting for results...")
            start_time = time.time()
            
            while time.time() - start_time < 300:  # 5 minute timeout
                async with session.get(f"{server_url}/get_output") as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        completed = result.get("completed_requests", [])
                        
                        if completed:
                            print("🎉 Results received!")
                            for req in completed:
                                print(f"Request ID: {req.get('request_id')}")
                                print(f"Status: {req.get('status')}")
                                if req.get('image'):
                                    print(f"Image saved: {req['image']}")
                            return True
                        else:
                            elapsed = int(time.time() - start_time)
                            print(f"   Still processing... ({elapsed}s elapsed)")
                
                await asyncio.sleep(3)
            
            print("Timeout - no results received")
            return False
            
    except aiohttp.ClientConnectorError:
        print(f"Cannot connect to {server_url}")
        print("   Make sure the server is running: python server.py")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
            description="Test client for SD3 server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Examples:
    python run_simple_example.py "A beautiful landscape"
    python run_simple_example.py "A cute cat" --timesteps 20
    python run_simple_example.py "Space station" --server-url http://192.168.1.100:8000
    python run_simple_example.py "Abstract art" --timesteps 25 --server-url http://localhost:8001
            """
        )
    
    parser.add_argument(
        "prompt",
        help="Text prompt for image generation"
    )
    
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=50,
        help="Number of denoising timesteps (default: 30)"
    )
    
    parser.add_argument(
        "--server-url", "-s",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.prompt.strip():
        print("Error: Prompt cannot be empty")
        sys.exit(1)
    
    if args.timesteps <= 0:
        print("Error: Timesteps must be positive")
        sys.exit(1)
    
    try:
        success = asyncio.run(test_server(args.prompt.strip(), args.server_url, args.timesteps))
        
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()