import os
import time
import logging
import pandas as pd
import requests
import sqlite3
import shutil
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

class ReputationAnalysis:
    def __init__(self,data_path: str):
        self.data_path = data_path

    def calculate(self):
        """
        Metodo che calola la reputation giornaliera 
        """
        logger.info("Starting calculation...")
        conn = sqlite3.connect(self.data_path)

        #giorno per giorno conto quanti tweet positivi, negativi, neutri
        df= pd.read_sql_query("""
        SELECT sentiment, date, COUNT(*) as count
        FROM tweets
        GROUP BY date, sentiment
        ORDER BY date, sentiment ASC
        """, conn)

        if df.empty:
            logger.info("No tweets to compute.")
            conn.close()
        else:
            logger.info(f"Found tweets to compute.")
            
            #creo la tabella reputation se non esiste
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reputation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                reputation_index FLOAT
            )
            """)

            #svuota la tabella reputation
            cursor.execute("DELETE FROM reputation")

            positive_count = 0;
            negative_count = 0;
            neutral_count = 0;

            current_date = None

            
            for index, row in df.iterrows():
                sentiment = row['sentiment']
                date = row['date']
                count = row['count']

                if current_date is None:
                    current_date = date
                elif current_date != date:
                    #calcolo reputation index
                    total = positive_count + negative_count + neutral_count
                    if total > 0:
                        reputation_index = (positive_count*2 + neutral_count*1 - negative_count*3) / total
                    else:
                        reputation_index = 0

                    #inserisco i dati nella tabella reputation
                    cursor.execute("""
                    INSERT INTO reputation (date, positive_count, negative_count, neutral_count, reputation_index)
                    VALUES (?, ?, ?, ?, ?)
                    """, (current_date, positive_count, negative_count, neutral_count, reputation_index))

                    #resetto i contatori
                    positive_count = 0
                    negative_count = 0
                    neutral_count = 0
                    current_date = date

                if sentiment == 'positive':
                    positive_count += count
                elif sentiment == 'negative':
                    negative_count += count
                elif sentiment == 'neutral':
                    neutral_count += count
                

            conn.commit()
            conn.close()

            logger.info("Updated reputation information in database.")

            #aggiorna db monitoraggio
            shutil.copy("../data/tweet.db","../monitoring/data/tweet.db/tweet.db")

        logger.info("Reputation analysis completed.")


if __name__ == "__main__":
    data_path="../data/tweet.db"
    reputation = ReputationAnalysis(data_path)
    reputation.calculate()