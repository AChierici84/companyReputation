from transformers import pipeline


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

        result=sentiment_task("The movie was okay, not the best I've seen but not the worst either.")
        assert result[0]['label'] == 'neutral'