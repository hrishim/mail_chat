import requests
import json
import logging
import time
import subprocess
import os
import sys
from typing import Optional, Dict, Any, NamedTuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ContainerConfig(NamedTuple):
    """Configuration for a container to test."""
    name: str
    image: str
    port: int
    model_name: str
    shm_size: str = "2g"  # Default shared memory size

class DockerContainerTester:
    # Default container configurations
    CONTAINER_CONFIGS = {
        "llama3": ContainerConfig(
            name="llama3-8b-instruct",
            image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
            port=8000,
            model_name="meta/llama3-8b-instruct"
        ),
        "llama3_1": ContainerConfig(
            name="llama3_1-8b-instruct",
            image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
            port=8001,
            model_name="meta/llama-3.1-8b-instruct"
        ),
        "deepseek-r1": ContainerConfig(
            name="deepseek-r1-distill-llama-8b",
            image="nvcr.io/nim/deepseek-ai/deepseek-r1-distill-llama-8b:1.5.2",
            port=8000,
            model_name="deepseek-ai/deepseek-r1-distill-llama-8b",
            shm_size="16GB"
        )
    }
    
    def __init__(self, log_file: Optional[str] = None, container_key: str = "llama3"):
        """Initialize the container tester with optional logging."""
        if container_key not in self.CONTAINER_CONFIGS:
            raise ValueError(f"Unknown container key: {container_key}. Valid keys are: {list(self.CONTAINER_CONFIGS.keys())}")
            
        self.config = self.CONTAINER_CONFIGS[container_key]
        self.logger = self._setup_logging(log_file) if log_file else None
        self.base_url = f"http://0.0.0.0:{self.config.port}"
    
    def _setup_logging(self, log_file: str) -> logging.Logger:
        """Set up logging to file."""
        logger = logging.getLogger('docker_test')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _log(self, message: str, level: str = 'info'):
        """Log message if logging is enabled."""
        if self.logger:
            getattr(self.logger, level)(message)
        print(message)
    
    def docker_login(self) -> bool:
        """Perform Docker login to NGC."""
        try:
            ngc_key = os.getenv('NGC_API_KEY')
            if not ngc_key:
                self._log("NGC_API_KEY environment variable not set", "error")
                return False
            
            result = subprocess.run(
                ["docker", "login", "nvcr.io", "--username", "$oauthtoken", "--password", ngc_key],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self._log("Docker login successful")
                return True
            else:
                self._log(f"Docker login failed: {result.stderr}", "error")
                return False
                
        except Exception as e:
            self._log(f"Docker login error: {str(e)}", "error")
            return False
    
    def get_container_status(self) -> str:
        """Get the current status of the container."""
        try:
            # First check if container is running
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.config.name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                return "stopped"
            
            status = result.stdout.strip().lower()
            if "up" not in status:
                return "stopped"
            
            # Container is up, now check if model is ready using health endpoint
            try:
                response = requests.get(f"{self.base_url}/v1/health/ready", timeout=2)
                self._log(f"Health endpoint status code: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    self._log(f"Health endpoint response: {data}")
                    if data.get("message") == "Service is ready.":
                        return "ready"
                return "starting"
            except (requests.exceptions.RequestException, ValueError) as e:
                # If health check fails or invalid JSON, model is still starting
                self._log(f"Health endpoint error: {str(e)}")
                return "starting"
                
        except Exception as e:
            self._log(f"Error checking container status: {str(e)}", "error")
            return "unknown"

    def wait_for_container_ready(self, timeout_seconds: int = 300, check_interval: int = 5) -> bool:
        """Wait for container to be fully ready with timeout."""
        self._log(f"Waiting up to {timeout_seconds} seconds for container to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_container_status()
            if status == "ready":
                self._log("Container is ready!")
                return True
            elif status == "stopped" or status == "unknown":
                self._log("Container failed to start", "error")
                return False
                
            self._log(f"Container status: {status}, waiting {check_interval} seconds...")
            time.sleep(check_interval)
            
        self._log(f"Container failed to become ready within {timeout_seconds} seconds", "error")
        return False
    
    def start_container(self) -> bool:
        """Start the container if it's not running."""
        try:
            if self.get_container_status() != "stopped":
                self._log("Container is already running")
                return True

            # Create NIM cache directory if it doesn't exist
            nim_cache = os.path.expanduser("~/.cache/nim")
            os.makedirs(nim_cache, exist_ok=True)
            os.chmod(nim_cache, 0o777)

            # First try to remove any stopped container with the same name
            subprocess.run(["docker", "rm", "-f", self.config.name], 
                         capture_output=True, text=True)
            
            # Ensure we're logged in
            if not self.docker_login():
                return False
            
            # Build docker command
            docker_cmd = [
                "docker", "run", "-d",
                "--name", self.config.name,
                "--gpus", "all",
                "-e", f"NGC_API_KEY={os.getenv('NGC_API_KEY')}",
                "-v", f"{nim_cache}:/opt/nim/.cache",
                "-u", str(os.getuid()),
                "-p", f"{self.config.port}:{self.config.port}",
                "--shm-size", self.config.shm_size,
                "--ulimit", "memlock=-1",
                "--ipc=host",
                self.config.image
            ]
            
            # Log the command (with API key redacted)
            log_cmd = docker_cmd.copy()
            for i, arg in enumerate(log_cmd):
                if arg.startswith("NGC_API_KEY="):
                    log_cmd[i] = "NGC_API_KEY=<redacted>"
            self._log(f"Running docker command: {' '.join(log_cmd)}")
            
            # Start the container
            self._log("Starting container...")
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self._log(f"Failed to start container: {result.stderr}", "error")
                return False
                
            self._log("Container started successfully")
            return True
            
        except Exception as e:
            self._log(f"Error starting container: {str(e)}", "error")
            return False
    
    def stop_container(self) -> bool:
        """Stop and remove the container."""
        try:
            self._log("Stopping container...")
            stop_result = subprocess.run(["docker", "stop", self.config.name], 
                                      capture_output=True, text=True)
            if stop_result.returncode != 0:
                self._log(f"Failed to stop container: {stop_result.stderr}", "error")
                return False
                
            self._log("Removing container...")
            rm_result = subprocess.run(["docker", "rm", self.config.name],
                                    capture_output=True, text=True)
            if rm_result.returncode != 0:
                self._log(f"Failed to remove container: {rm_result.stderr}", "error")
                return False
                
            self._log("Container stopped and removed successfully")
            return True
            
        except Exception as e:
            self._log(f"Error stopping container: {str(e)}", "error")
            return False
    
    def test_container_health(self) -> bool:
        """Test if the container is responding to basic health checks."""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                self._log("Container health check: OK")
                return True
            self._log(f"Container health check failed with status code: {response.status_code}", "error")
            return False
        except requests.exceptions.RequestException as e:
            self._log(f"Container health check failed with error: {str(e)}", "error")
            return False
    
    def list_available_models(self) -> list:
        """List all available models in the container."""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                self._log("Available models:")
                self._log(json.dumps(models, indent=2))
                return models
            self._log(f"Failed to list models. Status code: {response.status_code}", "error")
            return []
        except requests.exceptions.RequestException as e:
            self._log(f"Failed to list models: {str(e)}", "error")
            return []
    
    def test_model_inference(self) -> bool:
        """Test model inference with a simple prompt."""
        test_payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": "Write a limerick about the wonders of GPU computing."}],
            "max_tokens": 1024
        }
        
        try:
            self._log(f"Testing model inference for {self.config.model_name}")
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=test_payload
            )
            
            if response.status_code == 200:
                result = response.json()
                self._log("Model inference test successful")
                self._log("Response:")
                self._log(json.dumps(result, indent=2))
                return True
            
            self._log(f"Model inference test failed with status code: {response.status_code}", "error")
            self._log(f"Error response: {response.text}", "error")
            return False
            
        except requests.exceptions.RequestException as e:
            self._log(f"Model inference test failed with error: {str(e)}", "error")
            return False
    
    def run_all_tests(self, init_wait_seconds: int = 30) -> Dict[str, bool]:
        """Run all container tests and return results.
        
        Args:
            init_wait_seconds: Number of seconds to wait for container initialization
        """
        results = {
            "docker_login": False,
            "container_start": False,
            "container_ready": False,
            "health_check": False,
            "models_list": False,
            "inference": False,
            "cleanup": False
        }
        
        try:
            self._log("Starting container tests...")
            
            # First login to Docker
            results["docker_login"] = self.docker_login()
            if not results["docker_login"]:
                self._log("Docker login failed, skipping remaining tests", "error")
                return results
                
            # Start container if needed
            results["container_start"] = self.start_container()
            if not results["container_start"]:
                self._log("Container start failed, skipping remaining tests", "error")
                return results
                
            # Wait for container to initialize
            self._log(f"Waiting {init_wait_seconds} seconds for container to initialize...")
            time.sleep(init_wait_seconds)
            
            # Test container health
            results["health_check"] = self.test_container_health()
            if not results["health_check"]:
                self._log("Health check failed, skipping remaining tests", "error")
                return results
                
            # Container is ready if health check passed
            results["container_ready"] = True
            
            # Test models list
            models = self.list_available_models()
            results["models_list"] = len(models) > 0
            
            # Test model inference
            results["inference"] = self.test_model_inference()
            
            self._log("Test results summary:")
            self._log(json.dumps(results, indent=2))
            return results
            
        finally:
            # Always try to clean up, even if tests fail
            self._log("Cleaning up...")
            results["cleanup"] = self.stop_container()
            self._log("Final test results:")
            self._log(json.dumps(results, indent=2))


def test_all_containers(log_file: str = "docker_test.log") -> Dict[str, Dict[str, bool]]:
    """Test all configured containers in sequence."""
    results = {}
    
    for container_key in DockerContainerTester.CONTAINER_CONFIGS:
        print(f"\nTesting container: {container_key}")
        tester = DockerContainerTester(
            log_file=f"{container_key}_{log_file}",
            container_key=container_key
        )
        results[container_key] = tester.run_all_tests()
        
    return results


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test Docker containers for AI models")
    parser.add_argument("--container", type=str, help="Specific container key to test (default: test all containers)")
    parser.add_argument("--log-dir", type=str, default="test_results", help="Directory to store log files (default: test_results)")
    parser.add_argument("--init-wait", type=int, default=30, help="Seconds to wait for container initialization (default: 30)")
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get timestamp for log files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if args.container:
        # Test specific container
        if args.container not in DockerContainerTester.CONTAINER_CONFIGS:
            print(f"Error: Unknown container key '{args.container}'")
            print(f"Valid keys are: {list(DockerContainerTester.CONTAINER_CONFIGS.keys())}")
            sys.exit(1)
            
        print(f"\nTesting container: {args.container}")
        log_file = os.path.join(log_dir, f"{args.container}_test_{timestamp}.log")
        tester = DockerContainerTester(log_file=log_file, container_key=args.container)
        results = {args.container: tester.run_all_tests(init_wait_seconds=args.init_wait)}
    else:
        # Test all containers
        log_base = os.path.join(log_dir, f"container_test_{timestamp}")
        results = {}
        
        for container_key in DockerContainerTester.CONTAINER_CONFIGS:
            print(f"\nTesting container: {container_key}")
            log_file = f"{log_base}_{container_key}.log"
            tester = DockerContainerTester(log_file=log_file, container_key=container_key)
            results[container_key] = tester.run_all_tests(init_wait_seconds=args.init_wait)
    
    # Print summary
    print("\nOverall Test Results:")
    print(json.dumps(results, indent=2))