from src.train import train_pipeline
from src.model import save_artifact

TARGET = "Loan Status"
TRAIN_DATA_PATH = "Data/train.csv"

def run():
    print("Starting Loan Default Prediction Pipeline")

    trained_model, feature_names = train_pipeline(
        TRAIN_DATA_PATH,
        TARGET
    )

    save_artifact(trained_model, feature_names)

if __name__ == "__main__":
    run()