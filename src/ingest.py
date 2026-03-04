import pandas as pd
import json
import os
import re
from loguru import logger
from typing import List, Dict
try:
    from src.logging_config import setup_logger
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from src.logging_config import setup_logger

setup_logger()

class NustBankIngestor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.raw_excel = os.path.join(data_dir, "NUST Bank-Product-Knowledge.xlsx")
        self.raw_json = os.path.join(data_dir, "funds_transfer_app_features_faqFile.json")
        self.processed_data = []

    def sanitize_text(self, text: str) -> str:
        """Cleans text and anonymizes sensitive patterns."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        # Anonymize potential account numbers (PKXX NUST XXXX XXXX XXXX)
        text = re.sub(r'PK\d{2}\s?NUST\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}', '[NUST-ANNONYMIZED-ACCOUNT]', text)
        # Anonymize mobile numbers
        text = re.sub(r'(\+92|03)\d{9}', '[NUST-ANNONYMIZED-PHONE]', text)
        
        return text

    def ingest_json(self):
        logger.info(f"Starting JSON ingestion: {self.raw_json}")
        try:
            with open(self.raw_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for category_obj in data.get('categories', []):
                category = category_obj.get('category', 'General')
                for item in category_obj.get('questions', []):
                    question = self.sanitize_text(item.get('question'))
                    answer = self.sanitize_text(item.get('answer'))
                    
                    if question and answer:
                        self.processed_data.append({
                            "question": question,
                            "answer": answer,
                            "source": "JSON FAQ",
                            "category": category
                        })
            logger.success(f"Successfully ingested JSON. Total chunks: {len(self.processed_data)}")
        except Exception as e:
            logger.error(f"Failed to ingest JSON: {e}")

    def ingest_excel(self):
        logger.info(f"Starting Excel ingestion: {self.raw_excel}")
        try:
            xl = pd.ExcelFile(self.raw_excel)
            sheets = xl.sheet_names
            logger.info(f"Found {len(sheets)} sheets in Excel.")

            for sheet in sheets:
                if sheet == 'Main' or sheet == 'Sheet1':
                    continue
                
                logger.debug(f"Processing sheet: {sheet}")
                df = pd.read_excel(self.raw_excel, sheet_name=sheet)
                
                # Smart parsing logic for Question/Answer rows
                # Loop through the first column and look for '?' as questions
                # Assume next row or same row other column is answer
                
                rows = df.values.tolist()
                for i in range(len(rows)):
                    row_text = " ".join([str(val) for val in rows[i] if pd.notna(val)])
                    
                    if '?' in row_text:
                        question = self.sanitize_text(row_text)
                        
                        # Look at the next few rows for the answer
                        answer = ""
                        for j in range(i + 1, min(i + 3, len(rows))):
                            next_row_text = " ".join([str(val) for val in rows[j] if pd.notna(val)])
                            if '?' not in next_row_text and len(next_row_text) > 10:
                                answer = self.sanitize_text(next_row_text)
                                break
                        
                        if question and answer:
                            self.processed_data.append({
                                "question": question,
                                "answer": answer,
                                "source": f"Excel - {sheet}",
                                "category": sheet
                            })

            logger.success(f"Successfully ingested Excel. Current total chunks: {len(self.processed_data)}")
        except Exception as e:
            logger.error(f"Failed to ingest Excel: {e}")

    def save_processed(self):
        output_path = "data/processed/bank_knowledge_chunks.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, 'w') as f:
                json.dump(self.processed_data, f, indent=4)
            logger.success(f"Saved {len(self.processed_data)} chunks to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")

if __name__ == "__main__":
    ingestor = NustBankIngestor("Bank Knowledge")
    ingestor.ingest_json()
    ingestor.ingest_excel()
    ingestor.save_processed()
