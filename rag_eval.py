from chat_with_mail import EmailChatBot
import json
import time
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

def compute_metrics(prediction: str, reference: str):
    """
    Compute ROUGE-L and BLEU scores between prediction and reference
    """
    # ROUGE-L score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(prediction, reference)
    rouge_l = rouge_scores['rougeL'].fmeasure

    # BLEU score
    smoother = SmoothingFunction().method1
    prediction_tokens = nltk.word_tokenize(prediction.lower())
    reference_tokens = nltk.word_tokenize(reference.lower())
    bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoother)
    
    return {
        'rouge_l': rouge_l,
        'bleu': bleu,
        'average': (rouge_l + bleu) / 2
    }


def clean_model_response(response: str) -> str:
    """
    Clean up the model's response by removing any remaining system messages or UI prompts.
    """
    # Remove common UI prompts and system messages
    response = re.sub(r'System:|AI:|Human:|Assistant:|\[input response\]|Press enter to ask another question.*|What\'s your next question.*|You\'re welcome!.*', '', response)
    
    # Remove multiple newlines and spaces
    response = re.sub(r'\n+', ' ', response)
    response = re.sub(r'\s+', ' ', response)
    
    return response.strip()


def start_chat_bot() -> EmailChatBot:
    bot = EmailChatBot()
    print("Starting LLM container...")
    bot.start_container()
    time.sleep(60)  # Wait 1 minute
    print("Container should have started by now")
    return bot

def evaluate_rag(bot: EmailChatBot, rag_flow: str):
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        print("NLTK data downloaded successfully")

    # Load test cases
    with open('rag_test_cases.json', 'r') as f:
        test_data = json.load(f)
        test_cases = test_data['test_cases']
    
    total_scores = {'rouge_l': 0, 'bleu': 0, 'average': 0}
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
        print(f"Raw Model Answer: {model_answer}")
        
        # Clean up the model's response
        cleaned_answer = clean_model_response(model_answer)
        print(f"Cleaned Answer: {cleaned_answer}")
        print("--------------------------------")
        
        # Calculate metrics
        scores = compute_metrics(cleaned_answer.lower(), test_case['answer'].lower())
        print(f"ROUGE-L Score: {scores['rouge_l']:.3f}")
        print(f"BLEU Score: {scores['bleu']:.3f}")
        print(f"Average Score: {scores['average']:.3f}")
        
        # Update totals
        for metric, score in scores.items():
            total_scores[metric] += score

    # Calculate average scores
    print("\nFinal Scores:")
    for metric, total in total_scores.items():
        avg_score = total / num_cases
        print(f"Average {metric.upper()} Score: {avg_score:.3f}")

if __name__ == "__main__":
    bot = start_chat_bot()
    evaluate_rag(bot, "Conversational Chain")
    bot.stop_container()