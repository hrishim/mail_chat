#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
from container_manager import ContainerManager
from retrievers import QARetriever
from utils import setup_debug_logging, log_debug, log_error

def parse_args():
    parser = argparse.ArgumentParser(description='Test retrievers with various queries')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Directory to store test results')
    return parser.parse_args()

def wait_for_container(container_name: str, timeout: int = 300, check_interval: int = 20) -> bool:
    """Wait for container to be ready.
    
    Args:
        container_name: Name of the container to check
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        bool: True if container is ready, False if timeout reached
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = ContainerManager.check_container_status(container_name)
        if status == "ready":
            return True
        elif status == "error":
            log_error(f"Container {container_name} failed to start")
            return False
        
        log_debug(f"Container status: {status}, waiting {check_interval} seconds...")
        time.sleep(check_interval)
    
    log_error(f"Timeout waiting for container {container_name}")
    return False

def load_test_data() -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """Load test queries and answers from separate files.
    
    The template file is stored in git and contains queries.
    The answers file is local only and contains confidential expected answers.
    
    Returns:
        Tuple containing:
        - List of query dictionaries with 'id' and 'query' keys
        - Dictionary mapping query IDs to expected answers
    """
    # Load queries from template
    template_path = Path(__file__).parent / "test_queries_template.json"
    with open(template_path) as f:
        template_data = json.load(f)
    queries = template_data["queries"]
    
    # Try to load answers if available
    answers = {}
    answers_path = Path(__file__).parent / "test_queries_answers.json"
    if answers_path.exists():
        with open(answers_path) as f:
            answers_data = json.load(f)
        answers = answers_data["answers"]
    
    return queries, answers

def save_test_results(results: Dict[str, Any], output_dir: str):
    """Save test results to a JSON file."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"retriever_test_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def test_retrievers():
    """Run tests on retrievers with various queries."""
    args = parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup debug logging to file if enabled
    if args.debug:
        debug_log_path = output_dir / f"test_debug_{timestamp}.log"
        setup_debug_logging(args.debug, debug_log_path)
        print(f"Debug logs will be saved to: {debug_log_path}")
    else:
        setup_debug_logging(args.debug)
    
    # Start LLM container
    container_name = "meta-llama3-8b-instruct"
    ngc_key = os.getenv('NGC_API_KEY')
    if not ngc_key:
        log_error("NGC_API_KEY environment variable is required")
        return
    
    print("Starting LLM container...")
    ContainerManager.start_container(
        container_name=container_name,
        image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
        port=8000,
        ngc_key=ngc_key
    )
    
    # Wait for container to be ready
    if not wait_for_container(container_name):
        return
    
    try:
        # Initialize retriever
        retriever = QARetriever(
            user_name=os.getenv("USER_FULLNAME", "USER"),
            user_email=os.getenv("USER_EMAIL", "user@example.com"),
            debug_log=args.debug
        )
        
        # Load test data
        queries, answers = load_test_data()
        
        # Run tests
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": []
        }
        
        for query_data in queries:
            query_id = query_data["id"]
            query = query_data["query"]
            expected_answer = answers.get(query_id, "")
            
            print(f"\nTesting query: {query}")
            
            try:
                response = retriever.simple_qa(query)
                results["test_results"].append({
                    "query_id": query_id,
                    "query": query,
                    "response": response,
                    "expected_answer": expected_answer,
                    "success": True
                })
                print(f"Response: {response}")
                if expected_answer:
                    print(f"Expected: {expected_answer}")
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                log_error(error_msg)
                results["test_results"].append({
                    "query_id": query_id,
                    "query": query,
                    "error": error_msg,
                    "success": False
                })
        
        # Save results
        save_test_results(results, output_dir)
    
    finally:
        # Always stop container
        print("\nStopping LLM container...")
        ContainerManager.stop_container(container_name)

if __name__ == "__main__":
    test_retrievers()
