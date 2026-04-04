import json
import os
import random

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def augment_question(original_q):
    """
    Creates 4 additional conversational variations of the original question 
    to robustly train the LLM on different user intents and phrasings.
    """
    q_lower = original_q.strip().lower()
    
    # Strip trailing question marks for cleaner concatenation
    clean_q = original_q.strip()
    if clean_q.endswith('?'):
        clean_q = clean_q[:-1]
        
    variations = []
    
    # Template Set 1: Polite Inquiry
    prefixes_1 = ["Could you tell me", "I would like to know", "Can you explain", "Please tell me"]
    variations.append(f"{random.choice(prefixes_1)} {clean_q.lower()}?")
    
    # Template Set 2: Direct / Urgent
    prefixes_2 = ["Info needed:", "Help:", "Quick question:"]
    variations.append(f"{random.choice(prefixes_2)} {original_q}")
    
    # Template Set 3: Casual / Chatty
    prefixes_3 = ["Hey,", "Hi,", "Hello,"]
    suffixes_3 = ["- can you help?", "- any ideas?", "please."]
    variations.append(f"{random.choice(prefixes_3)} {clean_q.lower()} {random.choice(suffixes_3)}")
    
    # Template Set 4: Topic focus
    prefixes_4 = ["Regarding this:", "I have a query about this:"]
    variations.append(f"{random.choice(prefixes_4)} {original_q}")

    return variations

def main():
    input_path = "data/processed/bank_knowledge_chunks.json"
    output_path = "data/processed/bank_knowledge_chunks_augmented.json"
    
    print(f"Loading original dataset from {input_path}...")
    original_data = load_data(input_path)
    print(f"Original dataset size: {len(original_data)} chunks")
    
    augmented_dataset = []
    
    for item in original_data:
        # Keep original
        augmented_dataset.append(item)
        
        question = item.get("question", "")
        if not question:
            continue
            
        # Generate 4 variations
        variations = augment_question(question)
        
        for v_q in variations:
            new_item = item.copy()
            new_item["question"] = v_q
            augmented_dataset.append(new_item)
            
    print(f"Augmentation complete. New dataset size: {len(augmented_dataset)} chunks (~5x increase)")
    
    save_data(augmented_dataset, output_path)
    print(f"Saved augmented dataset to {output_path}")

if __name__ == "__main__":
    main()
