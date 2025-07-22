from prefect import flow, task
from classifier.pipeline.features import download_and_validate_data, extract_features
from classifier.pipeline.train import train_and_log_model


@task
def train_task(file_path):
    # ici tu peux renvoyer les métriques si tu veux
    return train_and_log_model(file_path)

@flow(name="music-genre-pipeline")
def music_genre_pipeline():
    data_path = download_and_validate_data()
    X, y = extract_features(data_path)
    train_task(data_path)

if __name__ == "__main__":
    music_genre_pipeline()

