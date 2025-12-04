import os
import logging
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
from configuration.config import Config
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

    def __init__(self):
        """
        Inizializza il SentimentTrainer con il modello e il tokenizer specificati.
        Args:
            model_path (str): Il percorso del modello pre-addestrato da utilizzare.
            max_length (int): La lunghezza massima per la tokenizzazione.
        """
        self.config = Config("./configuration/config.ini")
        self.model_path = self.config.get('training', 'model')
        self.max_length = int(self.config.get('training', 'max_length'))
        self.data_path = self.config.get('database', 'path')
        self.monitoring_path = self.config.get('database', 'monitoring_path')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=3)
        self.repo_name = self.config.get('training', 'repo_name')
        self.username = self.config.get('training', 'username')
 
        self.use_huggingface_hub = self.config.get('training', 'use_huggingface_hub')
        self.learning_rate = float(self.config.get('training', 'learning_rate'))
        self.batch_size = int(self.config.get('training', 'batch_size'))
        self.epochs = int(self.config.get('training', 'epochs'))
        self.weight_decay = float(self.config.get('training', 'weight_decay'))
        self.load_best_model = self.config.get('training', 'load_best_model')
        self.eval_strategy = self.config.get('training', 'eval_strategy')
        self.save_strategy = self.config.get('training', 'save_strategy')
        self.logging_steps = int(self.config.get('training', 'logging_steps'))
        self.dataloader_num_workers = int(self.config.get('training', 'dataloader_num_workers'))

        
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


    def train(self):
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

        label_mapping = {"negative": 0, "neutral": 1, "positive": 2, "0":0, "1": 1, "2": 2, "0.0": 0, "1.0": 1, "2.0": 2}
        
        df["label"] = df["user_feedback"].fillna(df["sentiment"])
        df["label"] = df["label"].astype(str).map(label_mapping)
        logger.info(df.head())
         
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

        # Sblocca solo gli ultimi 2 layer
        for param in model.roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Sblocca il classificatore finale
        for param in model.classifier.parameters():
            param.requires_grad = True

        tokenized_dataset = dataset.map(self.tokenize, batched=True)

        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        logger.info(tokenized_dataset["train"][0])

        # Configurazione dell'addestramento
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=self.repo_name, #directory di output
            eval_strategy=self.eval_strategy, # validaziane alla fine di ogni epoca
            save_strategy=self.save_strategy, # salvataggio del modello alla fine di ogni epoca
            learning_rate=self.learning_rate,   #learning_rate
            per_device_train_batch_size=self.batch_size,  #batch size training
            per_device_eval_batch_size=self.batch_size, #batch size validazione
            num_train_epochs=self.epochs, # numero di epoche
            weight_decay=self.weight_decay, #regolarizzazione (riduce i pesi del modello ogni step)
            load_best_model_at_end=self.load_best_model, #carichiamo il modello migliore
            logging_steps=self.logging_steps,  # ogni quanti step salvare i log
            dataloader_num_workers=self.dataloader_num_workers, #craicamento dati più veloce
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
        shutil.copy(self.data_path,self.monitoring_path)

        #se il modello non è abbastanza buono non lo salvo su hugging face
        if (results["eval_accuracy"]<0.9):
            logger.warning("Model regressing. Not pushing it.")
            return;
        

        # Salvataggio o caricamento del modello su Hugging Face Hub
        if self.use_huggingface_hub.lower() == 'true':

            login(token=os.getenv("HF_TOKEN")) # Effettua il login utilizzando il token di accesso
            model.push_to_hub(f"{self.username}/{self.repo_name}")
            tokenizer.push_to_hub(f"{self.username}/{self.repo_name}")
            dataset.push_to_hub(self.repo_name)
        else:
            trainer.save_model(self.repo_name)
            tokenizer.save_pretrained(self.repo_name)
            dataset.save_to_disk(self.repo_name)

        return results

if __name__ == "__main__":
    trainer = SentimentTrainer()
    results = trainer.train()
    print(results)
