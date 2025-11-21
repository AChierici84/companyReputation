import os
import logging
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
    def __init__(self, feedback_path: str):
        self.feedback_path = feedback_path
        self.dir=os.path.join("..","data","feedback")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def save_feedback(self):
        """
        Raccoglie feedback sulle previsioni incerte e le salva in un file.
        """
        logger.info("Starting feedback process of analyzed data...")
        # Carica i dati cercando csv in data_path
        for file in os.listdir(self.feedback_path):
            if file.endswith(".csv"):
                file_path = os.path.join(self.feedback_path, file)
                logger.info(f"Feedback on file: {file_path}")
                df = pd.read_csv(file_path)
                # filtra i tweet con confidenza sotto una soglia (es. 0.6)
                uncertain_df = df[df['confidence'] < 0.7]
                # chiedi feedback per ogni tweet incerto
                for index, row in uncertain_df.iterrows():
                    print(f"Tweet: {row['text']}")
                    print(f"Predicted sentiment: {row['sentiment']} with confidence {row['confidence']}")
                    feedback = input("Please provide the correct sentiment (NEGATIVE=0/NEUTRAL=1/POSITIVE=2): ")
                    uncertain_df.at[index, 'user_feedback'] = feedback.lower()
                # salva i risultati in un nuovo file csv
                output_file = os.path.join(self.dir, f"feedback_{file}")
                uncertain_df.to_csv(output_file, index=False)
                logger.info(f"Feedback saved to: {output_file}")

if __name__ == "__main__":
    feedback_path = os.path.join("..","data","analysis")
    feedback_manager = FeedbackManager(feedback_path)
    feedback_manager.save_feedback()
