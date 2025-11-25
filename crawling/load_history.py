import os
import sqlite3
import pandas as pd
import time

if __name__ == "__main__":
    data_path="../data/tweet.db"

    cnx = sqlite3.connect(data_path)

    df = pd.read_csv("../data/crawled/historic_tweets.csv") 
    df["id"] =
    df["author_id"] = df["author_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "sentiment" not in df.columns:
        df["sentiment"] = None
    if "confidence" not in df.columns:
        df["confidence"] = None
    if "user_feedback" not in df.columns:
        df["user_feedback"] = None

    df.to_sql(name='tweets', con=cnx, if_exists='append')
    print(f"Saved tweets to {data_path}")