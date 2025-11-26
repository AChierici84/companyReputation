import os
import logging
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
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
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, Features, ClassLabel, Value, DatasetDict
load_dotenv()

# Crea logs directory se non c'è
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
log_dir = os.path.abspath(log_dir)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'training.log')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class SentimentTrainer:

    def __init__(self, data_path: str, model_path: str, max_length: int = 128):
        """
        Inizializza il SentimentTrainer con il modello e il tokenizer specificati.
        Args:
            model_path (str): Il percorso del modello pre-addestrato da utilizzare.
            max_length (int): La lunghezza massima per la tokenizzazione.
        """
        self.model_path = model_path
        self.max_length = max_length
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
        
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

    def compute_metrics(self,eval_pred):
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

    def tokenize(self,batch):
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
            max_length=self.max_length
        )


    def train(self,push_to_hub=False):
        """
        Addestra un modello di sentiment analysis utilizzando il dataset tweet_eval.
        Args:
            model_path (str): Il percorso del modello pre-addestrato da utilizzare.
            push_to_hub (bool): Se True, il modello addestrato verrà caricato su Hugging Face Hub.
        Returns:
            dict: I risultati della valutazione del modello sul set di test.
        """

        
        conn = sqlite3.connect(self.data_path)

        set_seed(42)
        
        
        logger.info("Loading dataset...")


       # Leggiamo i dati
        df = pd.read_sql_query("""
            SELECT text, sentiment, user_feedback
            FROM tweets
            WHERE sentiment IS NOT NULL
        """, conn)

        if df.empty:
            print("No data available for training.")
            return
    

        df["label"] = df["user_feedback"].combine_first(df["sentiment"])
        label_mapping = {"negative": "negative", "neutral": "neutral", "positive": "positive", "0": "negative", "1": "neutral", "2": "positive", 0: "negative", 1: "neutral", 2: "positive"}
        df['label'] = df['label'].map(label_mapping)
         
        df_final = df[["text", "label"]].copy()

        # Shuffle e split
        hf_dataset = Dataset.from_pandas(df_final)
        feedback_dataset = hf_dataset.shuffle(seed=42)
        split_dataset = feedback_dataset.train_test_split(test_size=0.2)

        train_val_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

        split_dataset = train_val_dataset.train_test_split(test_size=0.1)

        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

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
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=3)
       
        #blocco i pesi del modello base
        for param in model.roberta.parameters():
            param.requires_grad = False

        tokenized_dataset = dataset.map(self.tokenize, batched=True)

        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Configurazione dell'addestramento
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="sentiment-roberta-finetuned", #directory di output
            eval_strategy="epoch", # validaziane alla fine di ogni epoca
            save_strategy="epoch", # salvataggio del modello alla fine di ogni epoca
            learning_rate=2e-5,   #learning_rate
            per_device_train_batch_size=4,  #batch size training
            per_device_eval_batch_size=4, #batch size validazione
            num_train_epochs=1, # numero di epoche
            weight_decay=0.01, #regolarizzazione (riduce i pesi del modello ogni step)
            load_best_model_at_end=True, #carichiamo il modello migliore
            logging_steps=50,  # ogni quanti step salvare i log
            dataloader_num_workers=2, #craicamento dati più veloce
            report_to=[]
        )

        # Creazione del Trainer
        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=self.compute_metrics
        )

        # Addestramento del modello
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed.")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #dump dei log di addestramento
        train_log = trainer.state.log_history
        log_dir = os.path.join("..", "data", "training")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, "train_log"+timestamp+".json")
        with open(log_file, "w") as f:
            import json
            json.dump(train_log, f, indent=4)
        logger.info(f"Training log saved to {log_file}")

        # Valutazione del modello
        logger.info("Evaluating model...")
        results=trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        logger.info(results)

        
        
        #salvo nel db il result 

        results_df = pd.DataFrame([{
            "start_time": start_time,
            "end_time": end_time,
            **results
        }])

        # Inserisci i nuovi risultati
        results_df.to_sql("training_results", conn, if_exists="append", index=False)
        conn.close()

        #aggiorna db monitoraggio
        shutil.copy("../data/tweet.db","../monitoring/data/tweet.db/tweet.db")

        # Salvataggio o caricamento del modello su Hugging Face Hub
        if push_to_hub:

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
    data_path="../data/tweet.db"
    trainer = SentimentTrainer(data_path=data_path,model_path="AChierici84/sentiment-roberta-finetuned", max_length=128)
    results = trainer.train(push_to_hub=True)
    print(results)
