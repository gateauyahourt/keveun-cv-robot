import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for MPS availability
if torch.backends.mps.is_available():
    print("MPS device is available. Using MPS for inference...")
    device = torch.device("mps")
else:
    print("MPS is not available. Using CPU for inference...")
    device = torch.device("cpu")

def generate_response(question, model, tokenizer, max_length=100):
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
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Loading fine-tuned model...")
    model_path = "./fine_tuned_model"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Set up special tokens
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    print("\nModel loaded successfully. Testing with sample questions:")
    print("-" * 50)
    
    test_questions = [
        "What is deep learning?",
        "How does machine learning work?",
        "What are neural networks used for?",
        "Explain artificial intelligence.",
        "What is supervised learning?"
    ]
    
    for question in test_questions:
        response = generate_response(question, model, tokenizer)
        print(f"\nQuestion: {question}")
        print(f"Generated Response: {response}")
        print("-" * 50)
    
    print("\nInteractive mode - type 'exit' to quit")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        
        response = generate_response(question, model, tokenizer)
        print(f"Generated Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
