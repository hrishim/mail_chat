import os
import subprocess
import requests
from typing import Optional
from utils import log_debug, args

class ContainerManager:
    """Class to manage Docker container operations."""
    
    @staticmethod
    def check_container_status(container_name: str, health_port: int = 8000) -> str:
        """Get the current status of a container.
        
        Args:
            container_name: Name of the Docker container to check
            health_port: Port number for the health endpoint (default: 8000 for LLM, use 8001 for reranker)
            
        Returns:
            str: Status of the container ('stopped', 'starting', or 'ready')
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
        """Start a Docker container.
        
        Args:
            container_name: Name to assign to the container
            image: Docker image to use
            port: Port to expose
            ngc_key: NGC API key if required (default: None)
            
        Returns:
            str: Status message
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
        """Stop a Docker container.
        
        Args:
            container_name: Name of the container to stop
            
        Returns:
            str: Status message
        """
        try:
            subprocess.run(["docker", "stop", container_name], check=True)
            subprocess.run(["docker", "rm", container_name], check=True)
            return "Container stopped"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error stopping container: {str(e)}")
            return f"Error stopping container: {str(e)}"
