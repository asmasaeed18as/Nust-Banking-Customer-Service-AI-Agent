import json
import os
import re
import random

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    # Remove random bullet artifacts and whitespace created from Excel parsers
    text = re.sub(r'^[o\-\·\•]\s*', '', text.strip())
    text = re.sub(r'\s+', ' ', text)
    # Fix standalone acronym fragments
    if text.lower() == "no initial deposit requirement":
        return "There is no initial deposit requirement for this account."
    if text.lower() == "attractive returns on savings account":
        return "This account offers attractive returns on your savings."
    if text.lower() == "medium enterprises":
        return "The target market for this account is Medium Enterprises."
    
    # Capitalize first letter safely
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text

def stylize_answer(answer):
    openings = [
        "", 
        "Certainly. ", 
        "I can help with that. ", 
        "To answer your question: ", 
        "Sure, here is the information: "
    ]
    closings = [
        "", 
        " Let me know if you need further details.", 
        " Is there anything else you'd like to check?"
    ]
    # Keep it helpful but not overly verbose (most times no opening/closing)
    op = random.choices(openings, weights=[60, 10, 10, 10, 10])[0]
    cl = random.choices(closings, weights=[80, 10, 10])[0]
    
    return f"{op}{answer}{cl}".strip()

def generate_ood_data():
    SYSTEM = "You are a helpful banking customer service assistant for NUST Bank."
    ood_queries = [
        "What is the weather like in Islamabad right now?",
        "Can you write a Python script for a binary search tree?",
        "Who won the last cricket world cup?",
        "Explain the theory of relativity to me.",
        "What's your opinion on current political elections?",
        "Translate 'how are you' into French.",
        "Generate a 5-day workout routine for me.",
        "Can you tell me a joke?",
        "How do I cook a perfect medium-rare steak?",
        "What is 250 multiplied by 984?"
    ]
    
    refusals = [
        "I specialize exclusively in NUST Bank's products and services. I'm afraid I cannot help with that.",
        "As an AI assistant for NUST Bank, I am unable to answer non-banking questions. How can I assist you with your banking needs?",
        "I'm sorry, I am programmed only to assist with NUST Bank related queries.",
        "I cannot provide assistance on that topic. Let me know if you need help with your NUST Bank account or services."
    ]
    
    ood_data = []
    for q in ood_queries:
        ood_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": random.choice(refusals)}
            ]
        })
    return ood_data

def generate_multiturn_data():
    SYSTEM = "You are a helpful banking customer service assistant for NUST Bank."
    # Crafting multi-turn conversations featuring coreference ("it", "this")
    conversations = [
        [
            ("What is the limit for mobile funds transfer?", "The daily transfer limit for the mobile app is 1 million."),
            ("Can I change it?", "Yes, you can manage and change your transfer limits in the 'My Profile' section of the mobile app under 'Manage Limit'.")
        ],
        [
            ("Do you have an account for kids?", "Yes, we offer the Little Champs Account which is designed specifically for minors under 18."),
            ("Does that account offer a debit card?", "Yes, the Little Champs Account comes with a free Debit Card and chequebook for the first issuance.")
        ],
        [
            ("Tell me about the NUST Waqaar Account.", "The NUST Waqaar Account is specially developed for senior citizens aged 55 and above."),
            ("What's the minimum balance for it?", "The minimum deposit requirement to open the Waqaar Savings account is Rs.1,000.")
        ],
        [
            ("How do I activate international transactions?", "You can enable international transactions by going to 'card management' in the app and selecting the 'international tranx activation' option."),
            ("Will I be charged for this?", "Please refer to the banking Schedule of Charges (SOC) for any applicable fees on international transactions.")
        ]
    ]
    
    mt_data = []
    for conv in conversations:
        messages = [{"role": "system", "content": SYSTEM}]
        for turn_q, turn_a in conv:
            messages.append({"role": "user", "content": turn_q})
            messages.append({"role": "assistant", "content": turn_a})
        mt_data.append({"messages": messages})
        
    return mt_data

def main():
    input_path = "data/processed/bank_knowledge_chunks.json"
    output_path = "data/processed/bank_knowledge_chunks_cooked.json"
    
    raw_data = load_data(input_path)
    
    final_dataset = []
    SYSTEM = "You are a helpful banking customer service assistant for NUST Bank."
    
    # 1. Clean and stylize base dataset
    for chunk in raw_data:
        q = clean_text(chunk.get("question", ""))
        a = clean_text(chunk.get("answer", ""))
        if not q or not a:
            continue
            
        a = stylize_answer(a)
        
        # Simulating a RAG context environment for finetuning
        # This forces the model to learn to extract from context instead of hallucinative memorization
        context = a # The answer itself acts as the mock retrieved context
        user_prompt = f"Context:\n{context}\n\nCustomer Query: {q}"
        
        final_dataset.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": a}
            ]
        })
        
    # 2. Add OOD (Out-of-domain) Rejections
    ood_data = generate_ood_data()
    final_dataset.extend(ood_data)
    
    # 3. Add Multi-Turn Data
    mt_data = generate_multiturn_data()
    final_dataset.extend(mt_data)
    
    print(f"Cooked dataset created. Total curated chunks: {len(final_dataset)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4)
        
if __name__ == "__main__":
    main()
