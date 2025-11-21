import os
import time
import logging
import pandas as pd
import requests
from dotenv import load_dotenv
import json
from requests.auth import AuthBase, HTTPBasicAuth

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#set logging file handler
file_handler = logging.FileHandler('../logs/crawler.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
API_KEY = os.getenv("API_KEY")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

class Crawler:
    def __init__(self):
        self.auth = HTTPBasicAuth(API_KEY, API_KEY_SECRET)
        self.search_url = "https://api.twitter.com/2/tweets/search/recent"
        self.bearerToken = BEARER_TOKEN

        # Parametri: start_time,end_time,since_id,until_id,max_results,next_token,
        # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
        self.query_params = {'query': '#KEY#','tweet.fields': 'author_id', 'max_results': '100', 'lang': 'en'}

    def get_bearer(self):
        """
        chiede il bearer token per l'autenticazione
        """
        url = "https://api.twitter.com/oauth2/token"
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
        with open("keywords.txt", "r") as f:
            keywords = f.readlines()
            for keyword in keywords:
                logger.info(f"Crawling tweets for keyword: {keyword.strip()}")
                keyword = keyword.strip()
                self.query_params['query'] = keyword

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

                # Salvare i tweet in un file csv
                if json_response['meta']['result_count'] > 0:
                    tweets = json_response['data']
                    df = pd.DataFrame(tweets)
                    filename = f"tweets_amazon.csv"
                    data_dir=os.path.join("..","data","crawled")
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    filename=os.path.join(data_dir,filename)

                    #se il file esiste già, aggiunge i nuovi tweet
                    if os.path.exists(filename):
                        df_existing = pd.read_csv(filename)
                        df = pd.concat([df_existing, df]).drop_duplicates(subset=['id'])
                    
                    df.to_csv(filename, index=False)

                    logger.info(f"Saved tweets to {filename}")

                    #salva il last id per riprendere il crawling in futuro
                    last_id = json_response['data'][-1]['id']
                    with open(os.path.join(data_dir,f"last_id.txt"), "w") as f:
                        f.write(last_id)
                    logger.info(f"Updated last id to: {last_id}")

                    #sleep per evitare di superare i limiti di twitter
                    time.sleep(5)
    


if __name__ == "__main__":
    crawler = Crawler()
    crawler.crawl()