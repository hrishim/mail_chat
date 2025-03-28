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
    
    # Base system prompt snippet (used across all methods)
    BASE_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about the user's email history. The user's name is {user_name} and their email address is {user_email}. When they ask questions using "I" or "me", it refers to {user_name} ({user_email})."""
    
    # Default system prompt template (for general use)
    DEFAULT_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + " Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep answers short and direct."
    
    def __init__(self, user_name, user_email, system_prompt=None, model_name="meta/llama3-8b-instruct"):
        """
        Initialize the prompt generator with user details, an optional system prompt, and model name.
        
        Args:
            user_name (str): The user's name.
            user_email (str): The user's email address.
            system_prompt (str, optional): Custom system prompt. If None, uses default.
            model_name (str, optional): The model name for the API (default: "meta/llama3-8b-instruct").
        """
        self.user_name = user_name
        self.user_email = user_email
        self.model_name = model_name
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
    
    def generate_api_prompt(self, user_query, context="", user_history=None):
        """
        Generate a prompt structure for NVIDIA NIM's OpenAI-compatible API, including user history.
        
        Args:
            user_query (str): The user's query.
            context (str, optional): Additional context for the query (default: empty string).
            user_history (list, optional): List of dicts with "user" and "assistant" keys (default: None).
        
        Returns:
            dict: A JSON-compatible dictionary with the messages array for the API.
        """
        # Initialize the messages list
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add user history (if provided)
        if user_history:
            for entry in user_history:
                if "user" in entry:
                    messages.append({"role": "user", "content": self._clean_text(entry["user"])})
                if "assistant" in entry:
                    messages.append({"role": "assistant", "content": self._clean_text(entry["assistant"])})
        
        # Add context as a separate user message (if provided)
        if context:
            cleaned_context = self._clean_text(context)
            messages.append({"role": "user", "content": f"Context: {cleaned_context}"})
        
        # Add user query
        cleaned_query = self._clean_text(user_query)
        messages.append({"role": "user", "content": cleaned_query})
        
        # Return the full payload
        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
    
    def generate_raw_prompt(self, user_query, context="", user_history=None):
        """
        Generate a raw text prompt with special tokens for direct model use, including user history.
        
        Args:
            user_query (str): The user's query.
            context (str, optional): Additional context for the query (default: empty string).
            user_history (list, optional): List of dicts with "user" and "assistant" keys (default: None).
        
        Returns:
            str: The fully formatted prompt with special tokens.
        """
        # Start with <|begin_of_text|>
        prompt = self.BEGIN_OF_TEXT
        
        # Add system prompt with role headers
        prompt += f"{self.START_HEADER}system{self.END_HEADER}{self.system_prompt}{self.EOT_ID}"
        
        # Add user history (if provided)
        if user_history:
            for entry in user_history:
                if "user" in entry:
                    prompt += f"{self.START_HEADER}user{self.END_HEADER}{self._clean_text(entry['user'])}{self.EOT_ID}"
                if "assistant" in entry:
                    prompt += f"{self.START_HEADER}assistant{self.END_HEADER}{self._clean_text(entry['assistant'])}{self.EOT_ID}"
        
        # Add context (if provided) with role headers
        if context:
            cleaned_context = self._clean_text(context)
            prompt += f"{self.START_HEADER}user{self.END_HEADER}Context: {cleaned_context}{self.EOT_ID}"
        
        # Add user query with role headers
        cleaned_query = self._clean_text(user_query)
        prompt += f"{self.START_HEADER}user{self.END_HEADER}{cleaned_query}{self.EOT_ID}"
        
        return prompt
    
    def summarize_prompt(self, text_to_summarize):
        """
        Generate an API prompt to summarize user-provided text (context or history).
        
        Args:
            text_to_summarize (str): The text to summarize (e.g., context or chat history).
        
        Returns:
            dict: A JSON-compatible dictionary with the messages array for the API.
        """
        system_prompt = self.BASE_SYSTEM_PROMPT.format(
            user_name=self.user_name, user_email=self.user_email
        ) + " Summarize the following text concisely."
        
        cleaned_text = self._clean_text(text_to_summarize)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text to summarize: {cleaned_text}"}
        ]
        
        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
    
    def decompose_prompt(self, user_query, context="", user_history=None):
        """
        Generate an API prompt to decompose a user question into multiple sub-questions.
        
        Args:
            user_query (str): The user's query to decompose.
            context (str, optional): Additional context for the query (default: empty string).
            user_history (list, optional): List of dicts with "user" and "assistant" keys (default: None).
        
        Returns:
            dict: A JSON-compatible dictionary with the messages array for the API.
        """
        system_prompt = self.BASE_SYSTEM_PROMPT.format(
            user_name=self.user_name, user_email=self.user_email
        ) + " Decompose the following question into multiple simpler questions based on the provided context and history. Return the sub-questions as a numbered list."
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add user history (if provided)
        if user_history:
            history_text = "Previous conversation:\n"
            for entry in user_history:
                if "user" in entry:
                    history_text += f"User: {self._clean_text(entry['user'])}\n"
                if "assistant" in entry:
                    history_text += f"Assistant: {self._clean_text(entry['assistant'])}\n"
            messages.append({"role": "user", "content": history_text})
        
        # Add context (if provided)
        if context:
            cleaned_context = self._clean_text(context)
            messages.append({"role": "user", "content": f"Context: {cleaned_context}"})
        
        # Add user query
        cleaned_query = self._clean_text(user_query)
        messages.append({"role": "user", "content": f"Question: {cleaned_query}"})
        
        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }

# Example usage
if __name__ == "__main__":
    # Initialize the prompt generator
    prompt_gen = Llama3PromptGenerator(user_name="Alice", user_email="alice@example.com")
    query = "What emails did I send and receive last week?"
    context = "The user has an email account with 50 emails from last week."
    user_history = [
        {"user": "Did I send any emails on Monday?", "assistant": "Yes, you sent 3 emails on Monday."},
        {"user": "What about Tuesday?", "assistant": "I don’t have that information."}
    ]

    # Example 1: API prompt with history
    api_prompt = prompt_gen.generate_api_prompt(query, context, user_history)
    print("Example 1 - API Prompt with History:")
    print(json.dumps(api_prompt, indent=2))
    
    # API call to local NIM
    try:
        response = requests.post(
            "http://0.0.0.0:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=api_prompt,
            timeout=30
        )
        if response.status_code == 200:
            print("Response:", response.json()["choices"][0]["message"]["content"])
        else:
            print("Error:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print("Request failed:", str(e))
    print()

    # Example 2: Raw prompt with history
    raw_prompt = prompt_gen.generate_raw_prompt(query, context, user_history)
    print("Example 2 - Raw Prompt with History:")
    print(raw_prompt)
    print()

    # Example 3: Summarize prompt
    text_to_summarize = "Alice sent 10 emails on Monday, received 15 on Tuesday, and had a meeting scheduled on Wednesday."
    summarize_prompt = prompt_gen.summarize_prompt(text_to_summarize)
    print("Example 3 - Summarize Prompt:")
    print(json.dumps(summarize_prompt, indent=2))
    print()

    # Example 4: Decompose prompt
    decompose_prompt = prompt_gen.decompose_prompt(query, context, user_history)
    print("Example 4 - Decompose Prompt:")
    print(json.dumps(decompose_prompt, indent=2))