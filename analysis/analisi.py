import os
import time
import logging
import pandas as pd
import requests
from datasets import Dataset
from dotenv import load_dotenv
from transformers import pipeline

# Crea logs directory se non c'è
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
log_dir = os.path.abspath(log_dir)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'analysis.log')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#set logging file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()

class Analisi:
    def __init__(self,data_path: str):
        self.data_path = data_path
        self.dir=os.path.join("..","data","analysis")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def analyze(self):
        """
        Metodo che analizza i tweet scaricati e salva i risultati in un file.
        """

        #scarica il mdoello corretto da huggingface
        model_path="AChierici84/sentiment-roberta-finetuned"

        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

        logger.info("Starting analysis of crawled data...")
        # Carica i dati cercando csv in data_path
        for file in os.listdir(self.data_path):
            if file.endswith(".csv"):
                file_path = os.path.join(self.data_path, file)
                logger.info(f"Analyzing file: {file_path}")
                df = pd.read_csv(file_path)
                # applica il sentiment analysis sui testi
                texts = df['text'].tolist()
                start_time = time.time()
                sentiments = sentiment_task(texts)
                end_time = time.time()
                logger.info(f"Sentiment analysis took {end_time - start_time} seconds")
                # aggiungi i risultati al dataframe
                df['sentiment'] = [sentiment['label'] for sentiment in sentiments]
                # salva i risultati in un nuovo file csv
                output_file = os.path.join(self.dir, f"analyzed_{file}")
                df.to_csv(output_file, index=False)
                logger.info(f"Saved analyzed data to {output_file}")

        logger.info("Analysis completed.")


if __name__ == "__main__":
    data_path=os.path.join("..","data","crawled")
    analisi = Analisi(data_path)
    analisi.analyze()