import os
import time
import logging
import pandas as pd
import requests
import sqlite3
import shutil
from configuration.config import Config
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
    def __init__(self):
        self.config = Config("./configuration/config.ini")
        self.data_path = self.config.get('database', 'path')
        self.model_path = self.config.get('analysis', 'model')
        self.monitoring_path = self.config.get('database', 'monitoring_path')

    def analyze(self):
        """
        Metodo che analizza i tweet scaricati e salva i risultati in un file.
        """

        #scarica il modello corretto da huggingface
        sentiment_task = pipeline("sentiment-analysis", model=self.model_path, tokenizer=self.model_path)

        logger.info("Starting analysis of crawled data...")
        conn = sqlite3.connect(self.data_path)

        df = pd.read_sql_query("""
        SELECT id, text
        FROM tweets
        WHERE sentiment IS NULL
        """, conn)

        if df.empty:
            logger.info("No new tweets to analyze.")
            conn.close()
        else:
            logger.info(f"Found {len(df)} tweets to analyze.")
            
            texts = df["text"].tolist()
            logger.info(texts)

            start_time = time.time()
            sentiments = sentiment_task(texts)
            end_time = time.time()

            logger.info(f"Sentiment analysis took {end_time - start_time} seconds")

            df["sentiment"] = [s["label"] for s in sentiments]
            df["confidence"] = [s["score"] for s in sentiments]

            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("""
                    UPDATE tweets
                    SET sentiment = ?, confidence = ?
                    WHERE id = ?
                """, (row["sentiment"], row["confidence"], row["id"]))

            conn.commit()
            conn.close()

            logger.info("Updated sentiment information in database.")

            #aggiorna db monitoraggio
            shutil.copy(self.data_path,self.monitoring_path)

        logger.info("Analysis completed.")


if __name__ == "__main__":
    analisi = Analisi()
    analisi.analyze()