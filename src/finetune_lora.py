import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from loguru import logger

try:
    from src.logging_config import setup_logger
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from src.logging_config import setup_logger

setup_logger()

# ======= CONFIGURATION =======
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "data/processed/bank_knowledge_chunks_cooked.json"
OUTPUT_DIR = "models/qwen_bank_lora"
MAX_SEQ_LENGTH = 512
# =============================

def load_and_prepare_data(data_path: str):
    """
    Loads JSON chunks and formats them into a huggingface Dataset
    with ChatML messaging format expected by instruction models.
    """
    logger.info(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = []
    
    for chunk in raw_data:
        # If the chunk already contains pre-formatted messages (e.g., for multi-turn or out-of-domain)
        if "messages" in chunk:
            formatted_data.append({"messages": chunk["messages"]})
            continue
            
        question = chunk.get("question", "")
        answer = chunk.get("answer", "")
        
        if not question or not answer:
            continue
            
        # We wrap the data in the conversational format that Qwen Instruct anticipates
        messages = [
            {"role": "system", "content": "You are a helpful banking customer service assistant for NUST Bank."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        formatted_data.append({"messages": messages})
        
    logger.success(f"Formatted {len(formatted_data)} conversational examples.")
    return Dataset.from_list(formatted_data)

def format_chat_prompt(example, tokenizer):
    """
    Applies the tokenizer's built-in chat template to format the string.
    """
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

def finetune():
    logger.info("Initializing QLoRA Fine-tuning pipeline")
    
    dataset = load_and_prepare_data(DATA_PATH)
    if dataset is None:
        return

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = dataset.map(lambda x: format_chat_prompt(x, tokenizer))

    # 2. Configure 4-bit Quantization (QLoRA) to save memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load Model
    logger.info(f"Loading Base Model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepares model for training (freezes original weights)
    model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA
    # We target the Q and V projection layers which usually yield the best adaptation results.
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    logger.info(f"Trainable Parameters Configured:")
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=200,                # Train for 200 steps (adjust based on dataset size/time)
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        optim="paged_adamw_8bit",     # Memory efficient optimizer
        bf16=True,
        run_name="nust_bank_qlora",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH
    )

    # 6. Initialize SFTTrainer
    logger.info("Starting Training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Execute Training
    trainer.train()

    # 8. Save the fine-tuned adapter
    logger.success(f"Training Complete! Saving local LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.success("Done! You can now load this adapter on top of the base Qwen model for inferences.")

if __name__ == "__main__":
    finetune()
