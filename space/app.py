import gradio as gr
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
    pipeline_model = get_pipeline()
    return pipeline_model(text)

# ------------------------
#      GRADIO UI
# ------------------------

def analyze(text):
    pipeline_model = get_pipeline()
    return pipeline_model(text)

demo = gr.Interface(
    fn=analyze,
    inputs="text",
    outputs="text",
    title="Company Sentiment Analyzer",
).queue()

# Mount Gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")


#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
