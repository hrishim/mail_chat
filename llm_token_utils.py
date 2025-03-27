from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import requests
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

#template = ChatPromptTemplate([
    #("system", "You are a helpful assistant."),
    #("human", "{input}")
#])

#prompt_value = template.invoke({"input": "Hello, world!"})
#print(prompt_value)
#print(type(prompt_value))

#print()

#template = PromptTemplate.from_template("Human: {input}\nAI: {output}")
#formatted_prompt = template.format(input="Hello", output="Hi there!")
#print(formatted_prompt)
#print(type(formatted_prompt))

def query_llm(prompt: str, max_tokens: int = 512) -> str:
    llm_url = "http://0.0.0.0:8000/v1/completions"
    try:
        response = requests.post(
            llm_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": "meta/llama3-8b-instruct",
                "prompt": prompt,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        response.raise_for_status()
        json_response = response.json()
        return json_response["choices"][0]["text"].strip()
    except requests.exceptions.Timeout:
        error_msg = "LLM request timed out"
        print(error_msg)
        return "I apologize, but the request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error to LLM"
        print(error_msg)
        return "I apologize, but I'm unable to connect to the LLM. Please try again later."
    except Exception as e:
        error_msg = f"Error querying LLM: {str(e)}"
        print(error_msg)
        return "I apologize, but I'm unable to process your request. Please try again later."

def test_llm():
    # Direct string prompt
    text = "Rephrase the following question and return only a single concise response: What is Gopal Srinivasan's email? Answer:"
    text = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an
    AI assistant that rephrases questions concisely without adding extra
    information.<|eot_id|><|start_header_id|>user<|end_header_id|> Rephrase the
    following question and return only a single concise response: What is Gopal
    Srinivasan's email?<|eot_id|><|start_header_id|>assistant<|end_header_id>
    """
    response = query_llm(text, max_tokens=127)
    print(f"Response: {response}")

def load_hf_token():
    """
    Loads the HF_TOKEN from a .env file or environment variable.
    
    Returns:
        str: The Hugging Face token, or None if not found.
    """
    # Load variables from .env file in the current directory
    load_dotenv()
    
    # Get HF_TOKEN from environment (either from .env or system env)
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env file or environment variables.")
    return hf_token

def print_special_tokens(model_id="meta-llama/Meta-Llama-3-8B-Instruct", 
hf_token=None):
    """
    Prints the special tokens for a given model tokenizer.
    
    Args:
        model_id (str): The Hugging Face model ID (default: "meta-llama/Meta-Llama-3-8B-Instruct").
        hf_token (str): Your Hugging Face API token (required for gated models like LLaMA 3).
    """
    from transformers import AutoTokenizer
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # Print special tokens map (named tokens like bos_token, eos_token)
        print("Special Tokens Map:")
        print(tokenizer.special_tokens_map)
        print()  # Blank line for readability
        
        # Print all special tokens in the vocabulary
        print("All Special Tokens:")
        print(tokenizer.all_special_tokens)
        print()
        
        # Print special tokens with their corresponding IDs
        print("Special Tokens with IDs:")
        for token in tokenizer.all_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{token}: {token_id}")
            
    except Exception as e:
        print(f"Error loading tokenizer or printing tokens: {e}")

def check_all_tokens(model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token=None):
    if hf_token is None:
        hf_token = load_hf_token()
    if not hf_token:
        print("Error: No valid HF_TOKEN found.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    # Print special tokens map and all special tokens
    print("Special Tokens Map:", tokenizer.special_tokens_map)
    print("All Special Tokens:", tokenizer.all_special_tokens)
    print()
    
    # Check specific LLaMA 3 tokens
    expected_tokens = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|no_sep|>"
    ]
    print("Checking Expected Tokens:")
    for token in expected_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            print(f"{token}: {token_id}")
        else:
            print(f"{token}: Not in vocabulary")

if __name__ == "__main__":
    # Automatically load token from .env or environment
    check_all_tokens()

