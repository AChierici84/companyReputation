import os
from transformers import pipeline
from dotenv import load_dotenv
import sqlite3
import pandas as pd
import numpy as np


class TestIntegration:
    """
    Classe di test di integrazione
    """
   
    def test_fine_tuned_model(self):
        model_path="AChierici84/sentiment-roberta-finetuned"

        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        result=sentiment_task("Lost communication with the driver? Lol guess @amazon got tired of me trying to figure out where my package that was supposed to be delivered yesterday is at. @AmazonHelp")
        assert result[0]['label'] == 'negative'

        result=sentiment_task("I love my new phone! The camera is amazing and the battery lasts all day.")
        assert result[0]['label'] == 'positive'

    def test_twitter_connection(self):
        from crawling.crawler import Crawler

        crawler = Crawler()
        bearer = crawler.get_bearer()
        assert len(bearer) > 0
        print("Bearer token retrieved successfully.")

    def test_hf_token(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        assert len(hf_token) > 0
        print("Hugging Face token retrieved successfully.")

    def test_api_keys(self):
        load_dotenv()
        api_key = os.getenv("API_KEY")
        api_key_secret = os.getenv("API_KEY_SECRET")
        bearer_token = os.getenv("BEARER_TOKEN")
        assert len(api_key) > 0
        assert len(api_key_secret) > 0
        assert len(bearer_token) > 0
        print("API keys retrieved successfully.")

    def test_tweet_sqlite_connection(self):
        data_path="./data/tweet.db"
        conn = sqlite3.connect(data_path)

        df = pd.read_sql_query("""
        SELECT id, text
        FROM tweets
        WHERE sentiment IS NULL
        """, conn)

        assert df.shape[0] >= 0
        assert df.shape[1] == 2

        conn.close()
    
    def test_training_sqlite_connection(self):
        data_path="./data/tweet.db"
        conn = sqlite3.connect(data_path)

        df = pd.read_sql_query("""
        SELECT start_time,end_time
        FROM training_results
        """, conn)

        assert df.shape[0] >= 0
        assert df.shape[1] == 2

        conn.close()

    def test_distribution_data(self):
        data_path="./data/tweet.db"
        conn = sqlite3.connect(data_path)

        df = pd.read_sql_query("""
        SELECT text, sentiment, user_feedback
            FROM tweets
            WHERE sentiment IS NOT NULL
        """, conn)

        assert df.shape[0] >= 0
        assert df.shape[1] == 3

        label_mapping = {"negative": 0, "neutral": 1, "positive": 2, "0":0, "1": 1, "2": 2, "0.0": 0, "1.0": 1, "2.0": 2}
        
        df["label"] = df["user_feedback"].fillna(df["sentiment"])
        df["label"] = df["label"].astype(str).map(label_mapping)
        assert df["label"].isna().sum() == 0

        count_test = len(df["label"])

        assert count_test >0

        unique_test, counts_test = np.unique(df["label"], return_counts=True)
        distribution = dict(zip(unique_test, counts_test))

        assert len(unique_test) > 1

        min_count = 15
        for label, count in distribution.items():
            assert count >= min_count, f"La classe {label} ha meno di {min_count} esempi"

        values = list(distribution.values())
        ratio = max(values) / min(values)
        assert ratio < 10, "Distribuzione sbilanciata"

        conn.close()
    


    
