import os
from transformers import pipeline
from dotenv import load_dotenv


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
    