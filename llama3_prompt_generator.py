import re
import json
import requests

class Llama3PromptGenerator:
    """
    A class to generate prompts for Meta-Llama-3-8B-Instruct, supporting both API and raw text formats.
    """
    
    # Special tokens for LLaMA 3 Instruct (used in raw prompt)
    BEGIN_OF_TEXT = "<|begin_of_text|>"
    EOT_ID = "<|eot_id|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    
    # Default system prompt template
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about the user's email history. The user's name is {user_name} and their email address is {user_email}. When they ask questions using "I" or "me", it refers to {user_name} ({user_email}). Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep answers short and direct."""
    
    def __init__(self, user_name, user_email, system_prompt=None):
        """
        Initialize the prompt generator with user details and an optional system prompt.
        
        Args:
            user_name (str): The user's name.
            user_email (str): The user's email address.
            system_prompt (str, optional): Custom system prompt. If None, uses default.
        """
        self.user_name = user_name
        self.user_email = user_email
        self.system_prompt = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT.format(
            user_name=self.user_name, user_email=self.user_email
        )
        # Clean the system prompt
        self.system_prompt = self._clean_text(self.system_prompt)
    
    def _clean_text(self, text):
        """
        Remove excess whitespace and normalize newlines in the text.
        
        Args:
            text (str): The text to clean.
        
        Returns:
            str: Cleaned text with single spaces and no leading/trailing whitespace.
        """
        return re.sub(r'\s+', ' ', text).strip()
    
    def generate_api_prompt(self, user_query, context=""):
        """
        Generate a prompt structure for NVIDIA NIM's OpenAI-compatible API.
        
        Args:
            user_query (str): The user's query.
            context (str, optional): Additional context for the query (default: empty string).
        
        Returns:
            dict: A JSON-compatible dictionary with the messages array for the API.
        """
        # Initialize the messages list
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context as a separate user message (if provided)
        if context:
            cleaned_context = self._clean_text(context)
            messages.append({"role": "user", "content": f"Context: {cleaned_context}"})
        
        # Add user query
        cleaned_query = self._clean_text(user_query)
        messages.append({"role": "user", "content": cleaned_query})
        
        # Return the full payload
        return {
            "model": "meta/llama3-8b-instruct",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
    
    def generate_raw_prompt(self, user_query, context=""):
        """
        Generate a raw text prompt with special tokens for direct model use.
        
        Args:
            user_query (str): The user's query.
            context (str, optional): Additional context for the query (default: empty string).
        
        Returns:
            str: The fully formatted prompt with special tokens.
        """
        # Start with <|begin_of_text|>
        prompt = self.BEGIN_OF_TEXT
        
        # Add system prompt with role headers
        prompt += f"{self.START_HEADER}system{self.END_HEADER}{self.system_prompt}{self.EOT_ID}"
        
        # Add context (if provided) with role headers
        if context:
            cleaned_context = self._clean_text(context)
            prompt += f"{self.START_HEADER}user{self.END_HEADER}Context: {cleaned_context}{self.EOT_ID}"
        
        # Add user query with role headers
        cleaned_query = self._clean_text(user_query)
        prompt += f"{self.START_HEADER}user{self.END_HEADER}{cleaned_query}{self.EOT_ID}"
        
        return prompt

# Example usage
if __name__ == "__main__":
    # Initialize the prompt generator
    prompt_gen = Llama3PromptGenerator(user_name="Alice", user_email="alice@example.com")
    query = "What emails did I send last week?"
    context = "The user has an email account with 50 emails from last week. They received 7 emails last week."

    # Example 1: API prompt
    api_prompt = prompt_gen.generate_api_prompt(query, context)
    print("Example 1 - API Prompt:")
    print(json.dumps(api_prompt, indent=2))
    
    # API call to local NIM (corrected)
    response = requests.post(
        "http://0.0.0.0:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=api_prompt,  # Pass the api_prompt dict directly
        timeout=30
    )
    if response.status_code == 200:
        print("Response:", response.json()["choices"][0]["message"]["content"])
    else:
        print("Error:", response.status_code, response.text)
    print()

    # Example 2: Raw prompt
    raw_prompt = prompt_gen.generate_raw_prompt(query, context)
    print("Example 2 - Raw Prompt:")
    print(raw_prompt)