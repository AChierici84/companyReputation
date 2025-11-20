from transformers import pipeline


class IntegrationTestClass:
    """
    Classe di test di integrazione
    """
    def test_base_model(self):
        model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"

        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        sentiment_task("Lost communication with the driver? Lol guess @amazon got tired of me trying to figure out where my package that was supposed to be delivered yesterday is at. @AmazonHelp")
    
    def test_fine_tuned_model(self):
        model_path="AChierici84/sentiment-roberta-finetuned"

        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        sentiment_task("Lost communication with the driver? Lol guess @amazon got tired of me trying to figure out where my package that was supposed to be delivered yesterday is at. @AmazonHelp")
    