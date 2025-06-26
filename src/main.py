from src.real_time_prediction import start_real_time_prediction
from src.model_training import train_model

if __name__ == '__main__':
    # Train the model (uncomment this line if you want to train it again)
    # train_model()

    # Start real-time webcam emotion prediction
    start_real_time_prediction()
