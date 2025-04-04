import os
import subprocess
import requests
from typing import Optional
from utils import log_debug, args

class ContainerManager:
    """A utility class for managing Docker containers running AI models.

    This class provides a high-level interface for managing Docker containers,
    specifically designed for AI model containers that expose health endpoints.
    It handles container lifecycle operations including:
    - Starting containers with proper GPU and cache configurations
    - Stopping and cleaning up containers
    - Monitoring container health and readiness states
    
    Features:
    - GPU support with proper NVIDIA configurations
    - NGC (NVIDIA GPU Cloud) authentication handling
    - Shared memory and IPC configurations for optimal performance
    - Health endpoint monitoring for model readiness
    - Automatic NIM cache directory management
    - Debug logging support
    
    Example Usage:
        ```python
        # Start an NGC container
        status = ContainerManager.start_container(
            container_name="llm-model",
            image="nvcr.io/nvidia/model:latest",
            port=8000,
            ngc_key="your-ngc-key"
        )
        
        # Check container status
        status = ContainerManager.check_container_status("llm-model")
        
        # Stop container when done
        status = ContainerManager.stop_container("llm-model")
        ```
    
    Note:
        - All methods are static for standalone usage
        - Requires Docker daemon to be running
        - GPU support requires NVIDIA Container Toolkit
        - Debug logging can be enabled via args.debugLog
    """
    
    @staticmethod
    def check_container_status(container_name: str, health_port: int = 8000) -> str:
        """Check the operational status of a Docker container and its model readiness.
        
        This method performs a two-step status check:
        1. Checks if the container is running using Docker commands
        2. If running, queries the container's health endpoint to verify model readiness
        
        Args:
            container_name: Name of the Docker container to check
            health_port: Port number for the health endpoint (default: 8000 for LLM, use 8001 for reranker)
            
        Returns:
            str: One of three possible states:
                - 'stopped': Container is not running
                - 'starting': Container is running but model is not ready
                - 'ready': Container is running and model is ready to serve
                
        Note:
            - Uses Docker CLI commands to check container status
            - Expects a health endpoint at /v1/health/ready
            - Health check has a 2-second timeout
            - Failed health checks return 'starting' to allow for model loading
        """
        try:
            # First check if container is running
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
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
                response = requests.get(f"http://0.0.0.0:{health_port}/v1/health/ready", timeout=2)
                if args.debugLog:
                    log_debug(f"Health endpoint status code: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    if args.debugLog:
                        log_debug(f"Health endpoint response: {data}")
                    if data.get("message") == "Service is ready.":
                        return "ready"
                return "starting"
            except (requests.exceptions.RequestException, ValueError) as e:
                # If health check fails or invalid JSON, model is still starting
                if args.debugLog:
                    log_debug(f"Health endpoint error: {str(e)}")
                return "starting"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Container status check error: {str(e)}")
            return "stopped"

    @staticmethod
    def start_container(container_name: str, image: str, port: int, 
                       ngc_key: Optional[str] = None) -> str:
        """Start a Docker container with proper configurations for AI model serving.
        
        This method handles the complete container startup process:
        1. Checks if container already exists/running
        2. Sets up NIM cache directory with proper permissions
        3. Removes any existing stopped container with same name
        4. Handles NGC authentication if required
        5. Starts container with optimal configurations for GPU usage
        
        Args:
            container_name: Name to assign to the container
            image: Docker image to use (supports NGC registry)
            port: Port to expose for model serving
            ngc_key: NGC API key for authenticated image pulls (default: None)
            
        Returns:
            str: Status message indicating:
                - Success: "Container starting..."
                - Already running: "Container is already running"
                - Error: Error message with details
                
        Container Configuration:
            - GPU access enabled with --gpus all
            - Shared memory size: 2GB
            - Unlimited memlock
            - Host IPC namespace
            - User permissions matched to host user
            - NIM cache mounted at /opt/nim/.cache
            
        Note:
            - Requires Docker daemon with NVIDIA Container Toolkit
            - NGC authentication is optional (only for NGC registry images)
            - Creates ~/.cache/nim directory if it doesn't exist
        """
        try:
            if ContainerManager.check_container_status(container_name) != "stopped":
                return "Container is already running"

            # Create NIM cache directory if it doesn't exist
            nim_cache = os.path.expanduser("~/.cache/nim")
            os.makedirs(nim_cache, exist_ok=True)
            os.chmod(nim_cache, 0o777)

            # First try to remove any stopped container with the same name
            subprocess.run(["docker", "rm", "-f", container_name], 
                         capture_output=True, text=True)
            
            # Login to NGC if key provided
            if ngc_key:
                if not ngc_key:
                    return "NGC_API_KEY environment variable not set"
                
                subprocess.run(["docker", "login", "nvcr.io", 
                              "--username", "$oauthtoken", 
                              "--password", ngc_key], check=True)

            # Start the container
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "--gpus", "all",
                "-v", f"{nim_cache}:/opt/nim/.cache",
                "-u", str(os.getuid()),
                "-p", f"{port}:{port}",
                "--shm-size=2g",
                "--ulimit", "memlock=-1",
                "--ipc=host"
            ]
            
            if ngc_key:
                cmd.extend(["-e", f"NGC_API_KEY={ngc_key}"])
                
            cmd.append(image)
            
            subprocess.run(cmd, check=True)
            
            return "Container starting..."
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error starting container: {str(e)}")
            return f"Error starting container: {str(e)}"

    @staticmethod
    def stop_container(container_name: str) -> str:
        """Stop and remove a Docker container.
        
        This method performs a clean container shutdown by:
        1. Stopping the running container
        2. Removing the container to clean up resources
        
        Args:
            container_name: Name of the container to stop
            
        Returns:
            str: Status message:
                - Success: "Container stopped"
                - Error: Error message with details
                
        Note:
            - Uses 'docker stop' followed by 'docker rm'
            - Safe to call on non-existent containers
            - Waits for container to stop gracefully
            - Removes container completely to avoid name conflicts
        """
        try:
            subprocess.run(["docker", "stop", container_name], check=True)
            subprocess.run(["docker", "rm", container_name], check=True)
            return "Container stopped"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error stopping container: {str(e)}")
            return f"Error stopping container: {str(e)}"
