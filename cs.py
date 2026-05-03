import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "SeriousDlqin2yrs"
ARTIFACT_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
WINSOR_COLS = ["DebtRatio", "MonthlyIncome"]
AGE_BINS = [0, 30, 40, 50, 60, 110]

RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000

# Параметры шкалы в скоркарте
PDO = 20
ODDS = 1 / 19
BASE_SCORE = 600
EPS = 1e-10


def load_data(path: str = "cs-training.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def fit_preprocess_params(df: pd.DataFrame) -> dict:
    df_fit = df[df["age"] > 0].copy()
    winsor_p99 = {col: df_fit[col].quantile(0.99) for col in WINSOR_COLS}
    median_dep = df_fit["NumberOfDependents"].median()
    monthly_income_median = df_fit["MonthlyIncome"].median()
    artifact_medians = {}
    for col in ARTIFACT_COLS:
        artifact_medians[col] = df_fit[col].replace(96, np.nan).median()

    return {
        "winsor_p99": winsor_p99,
        "median_dep": median_dep,
        "monthly_income_median": monthly_income_median,
        "artifact_medians": artifact_medians,
    }


def preprocess_data(
    df: pd.DataFrame, params: dict, is_train: bool = True
) -> pd.DataFrame:
    df_clean = df.copy()

    # Для train удаляем age <= 0, для test сохраняем все строки.
    if is_train:
        df_clean = df_clean[df_clean["age"] > 0]
    else:
        valid_age_median = df_clean.loc[df_clean["age"] > 0, "age"].median()
        df_clean.loc[df_clean["age"] <= 0, "age"] = valid_age_median

    # 2) Заменяем артефакт 96 на NaN
    for col in ARTIFACT_COLS:
        df_clean[col] = df_clean[col].replace(96, np.nan)

    # 3) Винзоризация выбросов
    df_clean["RevolvingUtilizationOfUnsecuredLines"] = df_clean[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1.0)
    for col in WINSOR_COLS:
        p99 = params["winsor_p99"][col]
        df_clean[col] = df_clean[col].clip(upper=p99)

    # 4) Импутация пропусков
    median_dep = params["median_dep"]
    df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].fillna(median_dep)

    df_clean["age_group"] = pd.cut(df_clean["age"], bins=AGE_BINS)
    df_clean["MonthlyIncome"] = df_clean.groupby("age_group", observed=True)[
        "MonthlyIncome"
    ].transform(lambda x: x.fillna(x.median()))
    df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].fillna(
        params["monthly_income_median"]
    )
    df_clean = df_clean.drop(columns="age_group")

    for col in ARTIFACT_COLS:
        df_clean[col] = df_clean[col].fillna(params["artifact_medians"][col])

    # 5) Инженерия признаков по просрочкам
    df_clean["TotalLatePayments"] = df_clean[LATE_COLS].sum(axis=1)
    df_clean = df_clean.drop(columns=LATE_COLS)
    return df_clean


def build_features_target(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df_clean.drop(columns=TARGET_COL)
    y = df_clean[TARGET_COL]
    return X, y


def split_and_scale(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train_scaled: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(
        C=LOGREG_C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=LOGREG_MAX_ITER,
    )
    model.fit(X_train_scaled, y_train)
    return model


def train_full_pipeline(
    X: pd.DataFrame, y: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = train_model(X_scaled, y)
    return model, scaler


def calculate_scorecard_points(y_pred_proba: np.ndarray) -> np.ndarray:
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(ODDS)
    print(f"Factor: {factor:.4f}")
    print(f"Offset: {offset:.4f}")

    y_pred_proba_safe = np.clip(y_pred_proba, EPS, 1 - EPS)
    log_odds = np.log((1 - y_pred_proba_safe) / y_pred_proba_safe)
    return offset + factor * log_odds


def print_score_stats(scores: np.ndarray, y_test: pd.Series) -> None:
    print("\nСтатистика скоров:")
    print(f"Минимум:  {scores.min():.0f}")
    print(f"Максимум: {scores.max():.0f}")
    print(f"Среднее:  {scores.mean():.0f}")
    print(f"Медиана:  {np.median(scores):.0f}")

    scores_series = pd.Series(scores, index=y_test.index)
    print(f"\nСредний скор надёжных заёмщиков: {scores_series[y_test == 0].mean():.0f}")
    print(f"Средний скор дефолтников:      {scores_series[y_test == 1].mean():.0f}")


def make_kaggle_submission(
    model: LogisticRegression,
    scaler: StandardScaler,
    params: dict,
    train_features: pd.DataFrame,
    test_path: str = "cs-test.csv",
    out_path: str = "submission.csv",
) -> None:
    df_test = load_data(test_path)
    df_test_clean = preprocess_data(df_test, params=params, is_train=False)
    df_test_clean = df_test_clean.reindex(columns=train_features.columns)

    X_submission = scaler.transform(df_test_clean)
    submission_proba = model.predict_proba(X_submission)[:, 1]

    submission = pd.DataFrame(
        {"Id": df_test_clean.index, "Probability": submission_proba}
    )
    submission.to_csv(out_path, index=False)

    print(f"\nСабмит сохранён в {out_path}")
    print(submission.head(10).to_string(index=False))
    print(f"\nСтрок в submission: {len(submission)}")


def main() -> None:
    df = load_data()
    preprocess_params = fit_preprocess_params(df)
    df_clean = preprocess_data(df, params=preprocess_params, is_train=True)

    X, y = build_features_target(df_clean)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)
    eval_model = train_model(X_train_scaled, y_train)

    y_pred_proba = eval_model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1
    print(f"AUC: {auc:.4f}")
    print(f"Gini: {gini:.4f}")

    coef_df = pd.DataFrame(
        {"feature": X.columns, "coefficient": eval_model.coef_[0]}
    )
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    print("\nТоп-5 признаков по модулю коэффициента:")
    print(coef_df.head(5).to_string(index=False))

    scores = calculate_scorecard_points(y_pred_proba)
    print_score_stats(scores, y_test)

    # Финальное обучение на всех train-данных для Kaggle сабмита.
    final_model, final_scaler = train_full_pipeline(X, y)
    make_kaggle_submission(
        final_model,
        final_scaler,
        preprocess_params,
        train_features=X,
        test_path="cs-test.csv",
        out_path="submission.csv",
    )


if __name__ == "__main__":
    main()
