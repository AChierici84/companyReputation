from fastapi import FastAPI
from transformers import pipeline


app = FastAPI(title="Company Reputation API")

# Carica modello
model_path="AChierici84/sentiment-roberta-finetuned"
sentiment_task = None

def get_pipeline():
    global sentiment_task
    if sentiment_task is None:
        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    return sentiment_task

@app.get("/")
def root():
    return {
        "status" : "OK",
        "message": "Company reputation alive! API documentation on /docs."
        }

@app.post("/predict")
def predict(text: str):
    pipeline = get_pipeline()
    result = pipeline(text)
    return result

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
