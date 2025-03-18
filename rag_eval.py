from chat_with_mail import EmailChatBot
import json
import time
from difflib import SequenceMatcher


def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def start_chat_bot() -> EmailChatBot:
    bot = EmailChatBot()
    print("Starting LLM container...")
    bot.start_container()
    time.sleep(60)  # Wait 1 minute
    print("Container should have started by now")
    return bot

def evaluate_rag(bot: EmailChatBot, rag_flow: str):
    # Load test cases
    with open('rag_test_cases.json', 'r') as f:
        test_data = json.load(f)
        test_cases = test_data['test_cases']
    
    total_similarity = 0
    num_cases = len(test_cases)

    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}/{num_cases}")
        print(f"Query: {test_case['query']}")
        print(f"Ground Truth: {test_case['answer']}")
        
        bot.clear_history()  # Clear history before each query
        if rag_flow == "Simple RAG":
            bot.chat_simple(test_case['query'], use_rerank=True)
        else:
            bot.chat_chain(test_case['query'], use_rerank=True)
        
        model_answer = bot.chat_history[-1]["content"]
        print(f"Model Answer: {model_answer}")
        
        # Calculate similarity between model answer and ground truth
        similarity = string_similarity(model_answer.lower(), test_case['answer'].lower())
        total_similarity += similarity
        print(f"Similarity Score: {similarity:.2f}")

    # Calculate average similarity
    avg_similarity = total_similarity / num_cases
    print(f"\nAverage Similarity Score: {avg_similarity:.2f}")

if __name__ == "__main__":
    bot = start_chat_bot()
    evaluate_rag(bot, "Simple RAG")
    bot.stop_container()