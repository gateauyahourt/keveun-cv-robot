import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU usage
print("Using CPU for inference...")
device = torch.device("cpu")

def generate_response(question, model, tokenizer, max_length=100):
    try:
        # Format the input
        input_text = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error generating the response."

def main():
    try:
        print("\n=== Starting Model Testing ===")
        print("\nLoading fine-tuned model...")
        
        model_path = "./fine_tuned_model"
        
        # Load model with optimized settings for CPU
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None  # Disable device mapping
        )
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set up special tokens
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
            response = generate_response(question, model, tokenizer)
            print(f"Question: {question}")
            print(f"Generated Response: {response}")
            print("-" * 50)
        
        print("\n=== Starting Interactive Mode ===")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break
            
            print("Generating response...")
            response = generate_response(question, model, tokenizer)
            print(f"Generated Response: {response}")
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
