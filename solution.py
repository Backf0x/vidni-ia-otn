import os
import sys
import time
import argparse
import joblib
import gc
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

class constants:
    TRAIN_FILENAME = "train.csv"
    TEST_FILENAME = "test.csv"
    USER_DATA_FILENAME = "users.csv"
    BOOK_DATA_FILENAME = "books.csv"
    BOOK_GENRES_FILENAME = "book_genres.csv"
    GENRES_FILENAME = "genres.csv"
    BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
    SUBMISSION_FILENAME = "submission.csv"
    TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
    PROCESSED_DATA_FILENAME = "processed_features.parquet"

    COL_USER_ID = "user_id"
    COL_BOOK_ID = "book_id"
    COL_TARGET = "rating"
    COL_SOURCE = "source"
    COL_PREDICTION = "rating_predict"
    COL_HAS_READ = "has_read"
    COL_TIMESTAMP = "timestamp"

    F_USER_MEAN_RATING = "user_mean_rating"
    F_USER_RATINGS_COUNT = "user_ratings_count"
    F_BOOK_MEAN_RATING = "book_mean_rating"
    F_BOOK_RATINGS_COUNT = "book_ratings_count"
    F_AUTHOR_MEAN_RATING = "author_mean_rating"
    F_BOOK_GENRES_COUNT = "book_genres_count"

    COL_GENDER = "gender"
    COL_AGE = "age"
    COL_AUTHOR_ID = "author_id"
    COL_PUBLICATION_YEAR = "publication_year"
    COL_LANGUAGE = "language"
    COL_PUBLISHER = "publisher"
    COL_AVG_RATING = "avg_rating"
    COL_GENRE_ID = "genre_id"
    COL_DESCRIPTION = "description"

    VAL_SOURCE_TRAIN = "train"
    VAL_SOURCE_TEST = "test"

    MISSING_CAT_VALUE = "-1"
    MISSING_NUM_VALUE = -1
    PREDICTION_MIN_VALUE = 0
    PREDICTION_MAX_VALUE = 10

class config:
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = ROOT_DIR / "output"
    MODEL_DIR = OUTPUT_DIR / "models"
    SUBMISSION_DIR = OUTPUT_DIR / "lgbm_tfidf_mae_earlystopping_mae"

    RANDOM_STATE = 42
    TARGET = constants.COL_TARGET

    TEMPORAL_SPLIT_RATIO = 0.8
    EARLY_STOPPING_ROUNDS = 50
    MODEL_FILENAME = "lgb_model.txt"

    TFIDF_MAX_FEATURES = 500
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    TFIDF_NGRAM_RANGE = (1, 2)

    CAT_FEATURES = [
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
        constants.COL_GENDER,
        constants.COL_AGE,
        constants.COL_AUTHOR_ID,
        constants.COL_PUBLICATION_YEAR,
        constants.COL_LANGUAGE,
        constants.COL_PUBLISHER,
    ]

    LGB_PARAMS = {
        "objective": "mae",
        "metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_STATE,
        "boosting_type": "gbdt",
    }

    LGB_FIT_PARAMS = {
        "eval_metric": "rmse",
        "callbacks": [],
    }

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != "category" and "datetime" not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    return df

def temporal_split_by_date(
    df: pd.DataFrame, split_date: pd.Timestamp, timestamp_col: str = constants.COL_TIMESTAMP
) -> Tuple[pd.Series, pd.Series]:
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    train_mask = df[timestamp_col] <= split_date
    val_mask = df[timestamp_col] > split_date
    return train_mask, val_mask

def get_split_date_from_ratio(
    df: pd.DataFrame, ratio: float, timestamp_col: str = constants.COL_TIMESTAMP
) -> pd.Timestamp:
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    sorted_timestamps = df[timestamp_col].sort_values()
    threshold_index = int(len(sorted_timestamps) * ratio)
    return sorted_timestamps.iloc[threshold_index]

def load_and_merge_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dtype_spec = {
        constants.COL_USER_ID: "int32",
        constants.COL_BOOK_ID: "int32",
        constants.COL_TARGET: "float32",
        constants.COL_GENDER: "category",
        constants.COL_AGE: "float32",
        constants.COL_AUTHOR_ID: "int32",
        constants.COL_PUBLICATION_YEAR: "float32",
        constants.COL_LANGUAGE: "category",
        constants.COL_PUBLISHER: "category",
        constants.COL_AVG_RATING: "float32",
        constants.COL_GENRE_ID: "int16",
    }
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_TARGET]},
        parse_dates=[constants.COL_TIMESTAMP],
    )
    train_df = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    test_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TEST_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_BOOK_ID]},
    )
    user_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.USER_DATA_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_GENDER, constants.COL_AGE]},
    )
    book_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_BOOK_ID, constants.COL_AUTHOR_ID, constants.COL_PUBLICATION_YEAR, constants.COL_LANGUAGE, constants.COL_AVG_RATING, constants.COL_PUBLISHER]},
    )
    book_genres_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_BOOK_ID, constants.COL_GENRE_ID]},
    )
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    book_descriptions_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME,
        dtype={constants.COL_BOOK_ID: "int32"},
    )
    train_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN
    test_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TEST
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    combined_df = combined_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    combined_df = combined_df.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")
    return combined_df, book_genres_df, genres_df, book_descriptions_df

def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [constants.COL_USER_ID, constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT]
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [constants.COL_BOOK_ID, constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT]
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")

def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [constants.COL_BOOK_ID, constants.F_BOOK_GENRES_COUNT]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")

def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME
    train_books = train_df[constants.COL_BOOK_ID].unique()
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")
    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
    description_map = dict(zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION]))
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")
    tfidf_matrix = vectorizer.transform(df_descriptions)
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)
    return pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    global_mean = train_df[config.TARGET].mean()
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)
    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)
    return df

def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    if include_aggregates:
        df = add_aggregate_features(df, train_df)
    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def prepare_data() -> None:
    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()
    featured_df = create_features(merged_df, book_genres_df, descriptions_df, include_aggregates=False)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")

def train() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}.")
    try:
        featured_df = pd.read_parquet(
            processed_path,
            engine="pyarrow",
            filters=[(constants.COL_SOURCE, '==', constants.VAL_SOURCE_TRAIN)]
        )
    except Exception:
        featured_df = pd.read_parquet(processed_path, engine="pyarrow")
        featured_df = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN]
    featured_df = reduce_mem_usage(featured_df)
    gc.collect()
    train_set = featured_df
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()
    del train_set, featured_df
    gc.collect()
    train_split_with_agg = add_aggregate_features(train_split, train_split)
    val_split_with_agg = add_aggregate_features(val_split, train_split)
    del train_split, val_split
    gc.collect()
    train_split_final = handle_missing_values(train_split_with_agg, train_split_with_agg)
    val_split_final = handle_missing_values(val_split_with_agg, train_split_with_agg)
    del train_split_with_agg, val_split_with_agg
    gc.collect()
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in train_split_final.columns if col not in exclude_cols]
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]
    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]
    del train_split_final, val_split_final
    gc.collect()
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=config.CAT_FEATURES, free_raw_data=True)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=config.CAT_FEATURES, reference=train_data, free_raw_data=True)
    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    model = lgb.train(
        config.LGB_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    val_preds = model.predict(X_val)
    val_preds = np.clip(val_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.save_model(str(model_path))

def predict() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}.")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()
    test_set_with_agg = add_aggregate_features(test_set.copy(), train_set)
    test_set_final = handle_missing_values(test_set_with_agg, train_set)
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in test_set_final.columns if col not in exclude_cols]
    non_feature_object_cols = test_set_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]
    X_test = test_set_final[features]
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model = lgb.Booster(model_file=str(model_path))
    test_preds = model.predict(X_test)
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)
    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME
    submission_df.to_csv(submission_path, index=False)

def validate() -> None:
    try:
        test_df = pd.read_csv(config.RAW_DATA_DIR / constants.TEST_FILENAME)
        sub_df = pd.read_csv(config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME)
        assert len(sub_df) == len(test_df)
        assert not sub_df[constants.COL_PREDICTION].isna().any()
        test_keys = test_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy().set_index([constants.COL_USER_ID, constants.COL_BOOK_ID])
        sub_keys = sub_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy().set_index([constants.COL_USER_ID, constants.COL_BOOK_ID])
        assert test_keys.index.equals(sub_keys.index)
        assert sub_df[constants.COL_PREDICTION].between(constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE).all()
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    if not (config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME).exists():
        prepare_data()
    train()
    predict()
    validate()