import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
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
from datasets import Dataset, Features, ClassLabel, Value
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

        logger.info("Loading dataset...")
        dataset = load_dataset("tweet_eval", "sentiment")
        retrain = False

        set_seed(42)

        #aggiungo al training set i dati di feedback se ci sono
        feedback_dir=os.path.join("..","data","feedback")
        if os.path.exists(feedback_dir):
           logger.info("Feedback directory found.")
           # Controlla se ci sono file di feedback recenti
           for file in os.listdir(feedback_dir):
               logger.info(f"Found feedback file: {file}")
               file_path = os.path.join(feedback_dir, file)
               # get file last edit
               ti_m = os.path.getmtime(file_path)
               last_edit = datetime.fromtimestamp(ti_m)
               logger.info(f"Last modified time: {last_edit}")
               if (datetime.now() - last_edit).days <= 1:
                   logger.info("Recent feedback data found.")
                   retrain = True
                   break
               else :
                   logger.info("No recent feedback data found.")
        if retrain:
            feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith(".csv")]
            if feedback_files:
                logger.info("updating training set with feedback data...")
                feedback_dfs = []
                for file in feedback_files:
                    file_path = os.path.join(feedback_dir, file)
                    logger.info(f"Loading feedback file: {file_path}")
                    df = pd.read_csv(file_path)
                    feedback_dfs.append(df)
                feedback_data = pd.concat(feedback_dfs, ignore_index=True)
                # Mappa i valori di user_feedback a etichette numeriche
                label_mapping = {"negative": "negative", "neutral": "neutral", "positive": "positive", "0": "negative", "1": "neutral", "2": "positive", 0: "negative", 1: "neutral", 2: "positive"}
                feedback_data['label'] = feedback_data['user_feedback'].map(label_mapping)
                feedback_data=feedback_data[['text', 'label']]
                logger.info(f"Feedback data shape: {feedback_data.shape}")
                logger.info(f"Feedback data sample:\n{feedback_data.head()}")
                # Crea un dataset Hugging Face dal DataFrame
                features = Features({
                    "text": Value("string"),
                    "label": ClassLabel(names=["negative", "neutral", "positive"])
                })
                feedback_dataset = dataset["train"].from_pandas(feedback_data[['text', 'label']],features=features)
                # Unisci i dataset
                dataset["train"] = concatenate_datasets([dataset["train"], feedback_dataset])
                logger.info(f"Added {len(feedback_dataset)} samples from feedback to training set.")
        else:
            logger.info("No retrain needed.")
            #exit the training function
            return {}


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
        tokenized_dataset = dataset.map(self.tokenize, batched=True)

        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Configurazione dell'addestramento
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="sentiment-roberta-finetuned",
            eval_strategy="no",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            weight_decay=0.01,
            load_best_model_at_end=False,
            logging_steps=100,
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

        #salvo in un csv start end e i result in un csv
        results_file = os.path.join(log_dir, "training_results.csv")

        results_df = pd.DataFrame([{
            "start_time": start_time,
            "end_time": end_time,
            **results
        }])
        if os.path.exists(results_file):
            df_existing = pd.read_csv(results_file)
            df = pd.concat([df_existing, results_df])
            results_df = df
        results_df.to_csv(results_file, index=False)
        logger.info(f"Training results saved to {results_file}")

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
    trainer = SentimentTrainer(model_path="AChierici84/sentiment-roberta-finetuned", max_length=128)
    results = trainer.train(push_to_hub=True)
    print(results)