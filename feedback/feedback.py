import os
import logging
import sqlite3
import pandas as pd

# Crea logs directory se non c'è
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
log_dir = os.path.abspath(log_dir)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'feedback.log')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#set logging file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class FeedbackManager:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.confidence_threshold=70

    def save_feedback(self):
        """
        Raccoglie feedback sulle previsioni incerte e le salva in un file.
        """
        logger.info("Starting feedback process of analyzed data...")
       
        conn = sqlite3.connect(self.data_path)

        # Leggi solo i tweet che hanno sentiment ma non hanno feedback, oppure bassa confidenza
        query = f"""
            SELECT id, text, sentiment, confidence, user_feedback
            FROM tweets
            WHERE sentiment IS NOT NULL and confidence IS NOT NULL 
            AND (confidence < {self.confidence_threshold} AND user_feedback IS NULL)
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            logger.info("No tweets need feedback.")
            conn.close()
            return

        for index, row in df.iterrows():
            print(f"Tweet: {row['text']}")
            print(f"Predicted sentiment: {row['sentiment']} with confidence {row['confidence']}")
            feedback = input("Please provide the correct sentiment (NEGATIVE=0/NEUTRAL=1/POSITIVE=2): ")
            while feedback not in ["0","1","2"]:
                feedback = input("Please provide the correct sentiment (NEGATIVE=0/NEUTRAL=1/POSITIVE=2): ")
            df.at[index, "user_feedback"] = int(feedback)

        # Aggiorna il DB con il feedback dell'utente
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                UPDATE tweets
                SET user_feedback = ?
                WHERE id = ?
            """, (row["user_feedback"], row["id"]))

        conn.commit()
        conn.close()

        logger.info("Feedback completed")            


if __name__ == "__main__":
    data_path = "../data/tweet.db"
    feedback_manager = FeedbackManager(data_path)
    feedback_manager.save_feedback()
