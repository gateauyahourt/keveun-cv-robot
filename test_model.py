import os
import readline # Allow to to use arrow keys to navigate history and edit the current line

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

# Pour cet exemple, on force l'utilisation du CPU
device = torch.device("cpu")

def generate_response_stream(question, model, tokenizer, max_length=200):
    """
    Génère une réponse en streaming à partir d'une question, en affichant les tokens au fur et à mesure.
    """
    try:
        # Préparer le texte d'entrée avec les tags
        input_text = f"Instruction: {question}\n<TAGS> professional\nResponse:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Initialiser le streamer qui va itérer sur les tokens générés
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Préparer les paramètres de génération en incluant le streamer
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_length": max_length,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "no_repeat_ngram_size": 2,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        # Lancer la génération dans un thread séparé pour pouvoir itérer sur le streamer
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        response = ""
        # Afficher en streaming chaque token généré
        for token in streamer:
            print(token, end="", flush=True)
            response += token
        
        thread.join()
        print()  # Retour à la ligne après la génération complète
        return response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error generating the response."

def main():
    try:
        print("\n=== Starting Model Testing ===")
        print("\nLoading fine-tuned model...")
        
        model_path = "./fine_tuned_model"
        
        # Charger le modèle avec des réglages optimisés pour le CPU
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None  # Désactive le mapping de device
        )
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configurer les tokens spéciaux
        print("Setting up tokens...")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        print("\nModel loaded successfully!")
        print("\nTesting with sample questions:")
        print("-" * 50)
        
        test_questions = [
            "Who is Kevin Manson?",
            "What is the Echo Forest?",
            "Who is Dr. Luna Starweaver?",
            "What is the Chronosphere Paradox?",
            "Who is Captain Nova Blackstar?",
            "What is the Stellar Harmony Theory?"
        ]
        
        for question in test_questions:
            print(f"\nProcessing question: {question}")
            response = generate_response_stream(question, model, tokenizer)
            print(f"\nQuestion: {question}")
            print(f"Generated Response: {response}")
            print("-" * 50)
        
        print("\n=== Starting Interactive Mode ===")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break
            
            print("Generating response (streaming)...")
            response = generate_response_stream(question, model, tokenizer)
            print(f"\nGenerated Response: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"\nError occurred during model testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        import traceback
        traceback.print_exc()