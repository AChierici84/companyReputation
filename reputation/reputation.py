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

class ReputationAnalysis:
    def __init__(self):
        self.config = Config("./configuration/config.ini")
        self.data_path = self.config.get('database', 'path')
        self.monitoring_path = self.config.get('database', 'monitoring_path')
        self.weights = [float(w) for w in self.config.get('reputation', 'weights').split(',')]

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
                        reputation_index = (positive_count*self.weights[0] + neutral_count*self.weights[1] - negative_count*self.weights[2]) / total
                        #round 2 decimals
                        reputation_index = round(reputation_index, 2)
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
            shutil.copy(self.data_path,self.monitoring_path)

        logger.info("Reputation analysis completed.")


if __name__ == "__main__":
    reputation = ReputationAnalysis()
    reputation.calculate()