import os
import logging
import numpy as np
import evaluate
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SentimentTrainer:

    def __init__(self, model_path: str, max_length: int = 128):
        """
        Inizializza il SentimentTrainer con il modello e il tokenizer specificati.
        Args:
            model_path (str): Il percorso del modello pre-addestrato da utilizzare.
            max_length (int): La lunghezza massima per la tokenizzazione.
        """
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        """
        Calcola le metriche di accuratezza e F1 score.
        Args:
            eval_pred (tuple): Una tupla contenente le logits del modello e le etichette vere.
        Returns:
            dict: Un dizionario contenente le metriche calcolate.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
                "accuracy": self.accuracy.compute(predictions=preds, references=labels)["accuracy"],
                "f1": self.f1.compute(predictions=preds, references=labels, average="macro")["f1"]
            }

    def tokenize(batch):
        """
        Tokenizza un batch di testi.
        Args:
            batch (dict): Un dizionario contenente una lista di testi sotto la chiave "text".
        Returns:
            dict: Un dizionario contenente gli input tokenizzati.
        """
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )


    def train(model_path,push_to_hub=False):
        """
        Addestra un modello di sentiment analysis utilizzando il dataset tweet_eval.
        Args:
            model_path (str): Il percorso del modello pre-addestrato da utilizzare.
            push_to_hub (bool): Se True, il modello addestrato verrà caricato su Hugging Face Hub.
        Returns:
            dict: I risultati della valutazione del modello sul set di test.
        """
        logger.info("Loading dataset...")
        dataset = load_dataset("tweet_eval", "sentiment")

        # Statistiche del dataset
        logger.info("Evaluating distribution...")
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

        logger.info(f"Train set: {count_train} samples, distribution: {distribution_train}")
        logger.info(f"Test set: {count_test} samples, distribution: {distribution_test}")
        logger.info(f"Validation set: {count_validation} samples, distribution: {distribution_validation}")
        logger.info(f"Total samples: {total}")

        # Tokenizzazione del dataset
        logger.info("Tokenizing dataset...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
        tokenized_dataset = dataset.map(tokenize, batched=True)

        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Configurazione dell'addestramento
        logger.info("Setting up training arguments...")
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

        # Creazione del Trainer
        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics
        )

        # Addestramento del modello
        logger.info("Starting training...")
        history=trainer.train()
        results=trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        logger.info(results)


        # Salvataggio o caricamento del modello su Hugging Face Hub
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

if __name__ == "__main__":
    trainer = SentimentTrainer("roberta-base")
    results = trainer.train(push_to_hub=True)
    print(results)