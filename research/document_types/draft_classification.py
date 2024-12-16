import logging
import pathlib
import dotenv
import os
import mlflow
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
import pickle as pkl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import toml

from research.lib import data_access, embeddings

REPOSITORY_ROOT = (pathlib.Path().cwd() / ".." / "..").resolve()

def remove_rows_with_missing_text(df: pd.DataFrame) -> pd.DataFrame:
    """ remove rows with missing text """
    empty_index = df["document_content_plain"] == ""
    empty_count = len(df[empty_index])
    print(f"Number of dropped empty texts: {empty_count} ({100 * empty_count / len(df_input):.1f}%)")
    return df.loc[~empty_index]

def create_embeddings(df: pd.DataFrame, column_to_embed: str, cache_directory: pathlib.Path = REPOSITORY_ROOT / "data" / "embeddings-cache"):
    embedding_model = embeddings.create_embedding_model(EMBEDDING_MODEL)
    mlflow.log_param("embedding_model.max_input_tokens", embedding_model.max_input_tokens)
    tokens = df[column_to_embed].progress_map(embedding_model.tokenize)
    with embeddings.use_cache(
            embedding_model,
            tqdm=tqdm,
            cache_directory=cache_directory,
    ) as get_embeddings:
        embeddings_doc_content_plain = get_embeddings(tokens.tolist())
        print(embeddings_doc_content_plain.shape)
    return embeddings_doc_content_plain

def save_model(model_file_name, classifier):
    model_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file_name, "wb") as f:
        pkl.dump(classifier, f)
        mlflow.log_artifact(model_file_name)

if __name__ == '__main__':

    dotenv.load_dotenv()
    config = toml.load("draft_classification.toml")

    DATA_FILE_NAME = config["data"]['data_file_name']
    PREPROCESSED_DATA_FILE = REPOSITORY_ROOT / "data" / "dataframes" / DATA_FILE_NAME
    DOCUMENT_SOURCES = config["data"]['document_sources']
    LANGUAGES = config["data"]['languages']
    FROM_YEAR = config["data"]['from_year']
    DOC_TYPES = config["data"]["doc_types"]
    EMBEDDING_MODEL = config["training"]['embedding_model']
    CV_FOLDS = config["training"]['cv_folds']
    TEST_SIZE = config["training"]['test_size']
    RANDOM_STATE = config["training"]['random_state']

    ### Logging parameters ###
    mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
    mlflow.set_experiment(experiment_name=f"Draft Classifier")
    if run := mlflow.active_run():
        logging.warning("Run = %s is already active, closing it.", run.info.run_name)
        mlflow.end_run()
    run = mlflow.start_run()
    print("Starting run:", run.info.run_name)
    mlflow.log_param("input_file", DATA_FILE_NAME)
    mlflow.log_param("document_sources", sorted(DOCUMENT_SOURCES))
    mlflow.log_param("languages", LANGUAGES)
    mlflow.log_param("from_year", FROM_YEAR)
    mlflow.log_param("doc_types", sorted(map(str, DOC_TYPES)))
    mlflow.log_param("embedding_model", EMBEDDING_MODEL)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.sklearn.autolog()
    tqdm.pandas()

    #### Load data ####
    data_access.ensure_dataframe_is_available(PREPROCESSED_DATA_FILE)
    df_input = data_access.load_consultation_documents(
        PREPROCESSED_DATA_FILE,
        only_document_sources=DOCUMENT_SOURCES,
        only_languages=LANGUAGES,
        only_doc_types=DOC_TYPES,
        starting_year=FROM_YEAR,
        mlflow=mlflow,
    )

    ### Preprocessing ###
    df_input = remove_rows_with_missing_text(df_input)
    # set target variable
    df_input.loc[:, "is_draft"] = (df_input["document_type"] == "DRAFT").astype(bool)
    # create embeddings
    embeddings_doc_content_plain = create_embeddings(df=df_input, column_to_embed="document_content_plain")

    #### Train-Test Split ####
    X = embeddings_doc_content_plain
    y = df_input["is_draft"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    mlflow.log_param("train_samples_count", len(X_train))
    mlflow.log_param("test_samples_count", len(X_test))
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    #### Model ####
    classifier = make_pipeline(
        StandardScaler(),
        # PCA(n_components=200, random_state=RANDOM_STATE),
        LogisticRegression(max_iter=1000),
        # SGDClassifier(loss="modified_huber", max_iter=1000),
        # GradientBoostingClassifier(random_state=RANDOM_STATE),
        # SVC(kernel="linear"),
    )

    #### CV ####
    cv_splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores_docs = cross_validate(
        classifier,
        X=X_train,
        y=y_train,
        cv=cv_splitter,
        scoring={
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
        }
    )
    avg_scores_docs = {k: np.round(np.mean(v), 4) for k, v in scores_docs.items()}
    mlflow.log_metrics({f"{k}_docs": v for k, v in avg_scores_docs.items() if k.startswith("test_")})

    print("Per-document CV scores:")
    print(f"Precision: {avg_scores_docs['test_precision']:.4f} (+/- {np.std(scores_docs['test_precision']):.4f})")
    print(f"Recall:    {avg_scores_docs['test_recall']:.4f} (+/- {np.std(scores_docs['test_recall']):.4f})")
    print(f"F1:        {avg_scores_docs['test_f1']:.4f} (+/- {np.std(scores_docs['test_f1']):.4f})")
    print(scores_docs)

    ## fit on the whole training set and evaluate on the test set ##
    classifier.fit(X_train, y_train)
    test_ground_truth_docs = y_test
    test_predictions_docs = classifier.predict(X_test)
    print("docs: ground truth", test_ground_truth_docs.shape)
    print("docs: predictions", test_predictions_docs.shape)

    print(classification_report(test_ground_truth_docs, test_predictions_docs))
    cm = confusion_matrix(test_ground_truth_docs, test_predictions_docs, normalize="true")
    print(cm)
    precision_recall_fscore_support(test_ground_truth_docs, test_predictions_docs)

    ## save model ##
    model_file = REPOSITORY_ROOT / "models" / "draft_classification.pkl"
    save_model(model_file_name=model_file, classifier=classifier)
