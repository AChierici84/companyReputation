from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_dataset

import numpy as np
import evaluate
from huggingface_hub import login
from dotenv import load_dotenv
import os



tokenizer = None
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

def tokenize(model_path,batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def train(model_path,push_to_hub=False):
    dataset = load_dataset("tweet_eval", "sentiment")
    count_train = len(dataset["train"])
    count_test = len(dataset["test"])
    count_validation = len(dataset["validation"])

    total=count_train+count_test+count_validation

    unique_train, counts_train = np.unique(dataset["train"]["label"], return_counts=True)
    distribution_train = dict(zip(unique_train, counts_train))

    unique_test, counts_test = np.unique(dataset["test"]["label"], return_counts=True)
    distribution_test = dict(zip(unique_test, counts_test))

    unique_validation, counts_validation = np.unique(dataset["validation"]["label"], return_counts=True)
    distribution_validation = dict(zip(unique_validation, counts_validation))

    print(f"Train set: {count_train} samples, distribution: {distribution_train}")
    print(f"Test set: {count_test} samples, distribution: {distribution_test}")
    print(f"Validation set: {count_validation} samples, distribution: {distribution_validation}")
    print(f"Total samples: {total}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenized_dataset = dataset.map(tokenize, batched=True)

    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="sentiment-roberta-finetuned", #directory di output
        eval_strategy="epoch", # validaziane alla fine di ogni epoca
        save_strategy="epoch", # salvataggio del modello alla fine di ogni epoca
        learning_rate=2e-5,   #learning_rate
        per_device_train_batch_size=16,  #batch size training
        per_device_eval_batch_size=16, #batch size validazione
        num_train_epochs=3, # numero di epoche
        weight_decay=0.01, #regolarizzazione (riduce i pesi del modello ogni step)
        load_best_model_at_end=True, #carichiamo il modello migliore
        logging_steps=50,  # ogni quanti step salvare i log
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics
    )

    history=trainer.train()
    print(history)
    results=trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(results)
    trainer.save_model("sentiment-roberta-finetuned")
    tokenizer.save_pretrained("sentiment-roberta-finetuned")

    if push_to_hub:
        load_dotenv()
        login(token=os.getenv("HF_TOKEN")) # Effettua il login utilizzando il token di accesso

        repo_name = "sentiment-roberta-finetuned"  # il nome che vuoi dare al modello sul tuo account
        username = "AChierici84"

        model.push_to_hub(f"{username}/{repo_name}")
        tokenizer.push_to_hub(f"{username}/{repo_name}")
        dataset.push_to_hub(repo_name)
    else:
        trainer.save_model("sentiment-roberta-finetuned")
        tokenizer.save_pretrained("sentiment-roberta-finetuned")
        dataset.save_to_disk("tweet_eval_sentiment_dataset")


    return results






