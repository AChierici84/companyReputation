import os
import sqlite3
import pandas as pd
from configuration.config import Config
import time
import random

def random_number(h):
    if h <= 0:
        raise ValueError("La lunghezza deve essere positiva")
    start = 10**(h-1)        # minimo numero con h cifre
    end = 10**h - 1          # massimo numero con h cifre
    return random.randint(start, end)



if __name__ == "__main__":
    config = Config("../configuration/config.ini")
    data_path = config.get('database', 'path')

    cnx = sqlite3.connect(data_path)

    df = pd.read_csv("../data/crawled/historic_tweets.csv") 
    df["id"] = [random_number(12) for _, row in df.iterrows()]
    df["author_id"] = [random_number(9) for _, row in df.iterrows()]
    df["edit_history_tweet_ids"] = ["['"+str(row["id"])+"']" for _, row in df.iterrows()]
    df["id"] = df["id"].astype(str)
    df["author_id"] = df["author_id"].astype(str)
    if "sentiment" not in df.columns:
        df["sentiment"] = None
    if "confidence" not in df.columns:
        df["confidence"] = None
    if "user_feedback" not in df.columns:
        df["user_feedback"] = None

    df= df[["id","edit_history_tweet_ids","author_id","text","sentiment","confidence","user_feedback","date"]]
    
    
    df.to_sql(name='tweets', con=cnx, if_exists='append', index=False)
    print(f"Saved tweets to {data_path}")

    cnx.close()