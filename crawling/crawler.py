import os
import sqlite3
import time
import logging
import pandas as pd
import requests
from configuration.config import Config
from dotenv import load_dotenv
import json
from requests.auth import AuthBase, HTTPBasicAuth

# Crea logs directory se non c'è
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
log_dir = os.path.abspath(log_dir)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'crawler.log')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#set logging file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()

class Crawler:
    def __init__(self):
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_KEY_SECRET")
        bearer_token = os.getenv("BEARER_TOKEN")

        if not api_key or not api_secret or not bearer_token:
            raise RuntimeError("Twitter API credentials not found in environment variables")
        self.config = Config("./configuration/config.ini")
        self.auth = HTTPBasicAuth(api_key, api_secret)
        self.search_url = self.config.get('crawler', 'search')
        self.oauth2_url = self.config.get('crawler', 'oauth2')
        self.bearerToken = bearer_token
        self.database= self.config.get('database', 'path')
        self.data_dir = self.config.get('crawler', 'data')
        self.last_id_file = self.config.get('crawler', 'last_id')
        self.keywords_file = self.config.get('crawler', 'keywords.file')

        # Parametri: start_time,end_time,since_id,until_id,max_results,next_token,
        # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
        self.query_params = {'query': '#KEY#','tweet.fields': 'author_id,created_at', 'max_results': '100'}

    def get_bearer(self):
        """
        chiede il bearer token per l'autenticazione
        """
        url = self.oauth2_url
        payload = {'grant_type': 'client_credentials'}
        response = requests.post(url, data=payload, auth=self.auth)
        if response.status_code != 200:
            raise Exception("Cannot get a Bearer token (HTTP {}): {}".format(response.status_code, response.text))
        r = response.json()
        return r.get("access_token")

    def bearer_oauth(self,r):
        """
        usa il bearer token per l'autenticazione delle richieste
        Args:
            r: richiesta da autenticare
        Returns:
            r: richiesta autenticata
        """

        r.headers["Authorization"] = f"Bearer {self.bearerToken}"
        r.headers["User-Agent"] = "v2RecentSearchPython"
        return r

    def connect_to_endpoint(self,url, params):
        """
        Connette all'endpoint di twitter e restituisce la risposta in formato json
        Args:
            url: endpoint di twitter
            params: parametri della richiesta
        Returns:
            json response: risposta in formato json
        """
        response = requests.get(url, auth=self.bearer_oauth, params=params)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


    def crawl(self):
        """
        Esegue il crawling dei tweet per le parole chiave specificate nel file keywords.txt
        """
        with open(self.keywords_file, "r") as f:
            keywords = f.readlines()
            for keyword in keywords:
                logger.info(f"Crawling tweets for keyword: {keyword.strip()}")
                keyword = keyword.strip()
                self.query_params['query'] = keyword+" lang:en -is:retweet"

                #se trova un file con last id, lo usa per riprendere il crawling
                data_dir=os.path.join("data","crawled")
                last_id_file=os.path.join(data_dir,"last_id.txt")
                if os.path.exists(last_id_file):
                    with open(last_id_file, "r") as f:
                        last_id = f.read().strip()
                        self.query_params['since_id'] = last_id
                        logger.info(f"Resuming from last id: {last_id}")
                
                # Effettua la richiesta all'endpoint
                json_response = self.connect_to_endpoint(self.search_url, self.query_params)
                print(json.dumps(json_response, indent=4, sort_keys=True))
                logger.info(f"Found results: {json_response['meta']['result_count']} tweets")

                sql_database_path=self.database

                # Salvare i tweet in un db sqlite
                if json_response['meta']['result_count'] > 0:
                    tweets = json_response['data']
                    df = pd.DataFrame(tweets)
                    df["id"] = df["id"].astype(str)
                    if "author_id" in df.columns:
                        df["author_id"] = df["author_id"].astype(str)
                    if "created_at" in df.columns:
                        df.rename(columns={"created_at": "date"}, inplace=True)
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    if "sentiment" not in df.columns:
                        df["sentiment"] = None
                    if "confidence" not in df.columns:
                        df["confidence"] = None
                    if "user_feedback" not in df.columns:
                        df["user_feedback"] = None
                    cnx = sqlite3.connect(sql_database_path)
                    df.to_sql(name='tweets', con=cnx, if_exists='append', index=False)
                    logger.info(f"Saved tweets to {sql_database_path}")

                    #salva il last id per riprendere il crawling in futuro
                    last_id = json_response['data'][-1]['id']
                    with open(self.last_id_file, "w") as f:
                        f.write(last_id)
                    logger.info(f"Updated last id to: {last_id}")

                    #sleep per evitare di superare i limiti di twitter
                    time.sleep(5)
    


if __name__ == "__main__":
    crawler = Crawler()
    crawler.crawl()