"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

ARM_NAMES = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
ARM_TO_IDX = {name: i for i, name in enumerate(ARM_NAMES)}

FEATURES_CAT = ["zip_code", "channel", "history_segment"]
FEATURES_NUM = ["recency", "log_history", "mens", "womens", "newbie"]
ALL_BASE_FEATURES = FEATURES_NUM + FEATURES_CAT

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_history"] = np.log1p(df["history"].clip(lower=0))
    return df


def preprocess_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = add_features(train_df.copy())
    test_df = add_features(test_df.copy())

    # numeric
    num_medians = train_df[FEATURES_NUM].median()
    train_df[FEATURES_NUM] = train_df[FEATURES_NUM].fillna(num_medians)
    test_df[FEATURES_NUM] = test_df[FEATURES_NUM].fillna(num_medians)

    # categorical
    cat_modes = {}
    for col in FEATURES_CAT:
        mode = train_df[col].astype(str).mode(dropna=True)
        if len(mode) > 0:
            cat_modes[col] = mode.iloc[0]
        else:
            cat_modes[col] = "UNKNOWN"

    for df in (train_df, test_df):
        for col in FEATURES_CAT:
            df[col] = df[col].astype(str).fillna(cat_modes[col])

    return train_df, test_df

# S-LEARNER 

def fit_single_s_learner(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    a_tr: pd.Series,
    random_seed: int = 42,
    depth: int = 6,
    learning_rate: float = 0.05,
    n_estimators: int = 700
):
    """
    Один CatBoost: visit ~ x + arm_name
    c заданным сидом и гиперпарами.
    """
    X_s = X_tr.copy()
    X_s["arm_name"] = a_tr.astype(str)

    feature_cols = list(X_s.columns)
    cat_features = FEATURES_CAT + ["arm_name"]
    cat_features_idx = [feature_cols.index(c) for c in cat_features]

    pos_rate = y_tr.mean()
    if pos_rate <= 0 or pos_rate >= 1:
        class_weights = [1.0, 1.0]
    else:
        w_pos = (1 - pos_rate) / pos_rate
        w_pos = float(min(max(w_pos, 1.0), 6.0))
        class_weights = [1.0, w_pos]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        depth=depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_seed=random_seed,
        verbose=False,
        class_weights=class_weights
    )

    model.fit(
        X_s[feature_cols],
        y_tr,
        cat_features=cat_features_idx
    )

    return model, feature_cols


def predict_q_s(model_s,
                X: pd.DataFrame,
                feature_cols_s: list) -> np.ndarray:
    """
    q_S: (N,3) — P(visit | x, arm) из S-learner.
    """
    N = len(X)
    q = np.zeros((N, len(ARM_NAMES)), dtype=float)

    for j, arm_name in enumerate(ARM_NAMES):
        X_aug = X.copy()
        X_aug["arm_name"] = arm_name
        X_aug = X_aug[feature_cols_s]
        q[:, j] = model_s.predict_proba(X_aug)[:, 1]

    return q

# ENSEMBLE OF S-LEARNERS


def fit_s_ensemble(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    a_tr: pd.Series,
    seeds=(42, 43, 44, 45, 46),
    depth: int = 6,
    learning_rate: float = 0.05,
    n_estimators: int = 700
):
    """
    Обучаем ансамбль S-learner моделей с разными сидом.
    Возвращаем список (model, feature_cols).
    """
    ensemble = []
    for seed in seeds:
        print(f"Training S-learner with seed={seed} ...")
        model_s, feature_cols_s = fit_single_s_learner(
            X_tr, y_tr, a_tr,
            random_seed=seed,
            depth=depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators
        )
        ensemble.append((model_s, feature_cols_s))
    return ensemble


def predict_q_ensemble(
    ensemble,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Усредняем q_S по ансамблю.
    """
    N = len(X)
    q_sum = np.zeros((N, len(ARM_NAMES)), dtype=float)

    for model_s, feature_cols_s in ensemble:
        q_sum += predict_q_s(model_s, X, feature_cols_s)

    q_mean = q_sum / len(ensemble)
    return q_mean


# POLICY & SNIPS

def build_policy_eps_greedy(q: np.ndarray, eps: float) -> np.ndarray:
    """
    ε-greedy политика вокруг argmax(q).
    """
    N, K = q.shape
    best = q.argmax(axis=1)
    pi = np.full((N, K), eps, dtype=float)
    pi[np.arange(N), best] = 1.0 - eps * (K - 1)
    return pi


def snips_score(pi: np.ndarray,
                a_val: pd.Series,
                y_val: pd.Series,
                logging_prop: float = 1.0 / 3.0) -> float:
    arm_indices = np.array([ARM_TO_IDX[a] for a in a_val])
    pi_chosen = pi[np.arange(len(y_val)), arm_indices]
    w = pi_chosen / logging_prop

    numer = (w * y_val.values).sum()
    denom = w.sum()
    if denom == 0:
        return 0.0
    return numer / denom


def tune_eps_for_ensemble(q_val: np.ndarray,
                          a_val: pd.Series,
                          y_val: pd.Series,
                          eps_grid=None):
    """
    Подбираем eps для ε-greedy по SNIPS на ансамблевом q.
    """
    if eps_grid is None:
        eps_grid = [0.0, 0.001, 0.002, 0.003, 0.004]

    best_eps = None
    best_score = -1e9

    print("Tuning eps for ensemble (ε-greedy):")
    for eps in eps_grid:
        pi_val = build_policy_eps_greedy(q_val, eps)
        score = snips_score(pi_val, a_val, y_val)
        print(f"  eps={eps:.3f} -> SNIPS={score:.6f}")
        if score > best_score:
            best_score = score
            best_eps = eps

    print(f"\nBest eps for ensemble: eps={best_eps:.3f}, SNIPS={best_score:.6f}\n")
    return best_eps, best_score

def train_and_predict(
    train_path: str = "train.csv",
    test_path: str = "test.csv",
    submission_path: str = "submission.csv",
    random_state: int = 42
):
    set_seeds(random_state)

    # 1. load & preprocess
    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = preprocess_train_test(train_df, test_df)

    X_all = train_df[ALL_BASE_FEATURES]
    y_all = train_df["visit"]
    a_all = train_df["segment"]

    # 2. train/val split
    X_tr, X_val, y_tr, y_val, a_tr, a_val = train_test_split(
        X_all,
        y_all,
        a_all,
        test_size=0.3,
        random_state=random_state,
        stratify=a_all
    )

    # 3. обучаем ансамбль S-learner на train-части
    ensemble_seeds = (42, 43, 44, 45, 46)
    s_ensemble = fit_s_ensemble(
        X_tr, y_tr, a_tr,
        seeds=ensemble_seeds,
        depth=6,
        learning_rate=0.05,
        n_estimators=700
    )

    # 4. считаем ансамблевый q на валидации
    q_val_ens = predict_q_ensemble(s_ensemble, X_val)

    # 5. тюним eps для ансамбля
    eps_best, snips_best = tune_eps_for_ensemble(q_val_ens, a_val, y_val)

    print(f"Final choice on validation: eps={eps_best:.3f}, SNIPS={snips_best:.6f}")

    # 6. обучаем ансамбль на ВСЁМ train_full
    s_ensemble_full = fit_s_ensemble(
        X_all, y_all, a_all,
        seeds=ensemble_seeds,
        depth=6,
        learning_rate=0.05,
        n_estimators=700
    )

    # 7. строим политику на test
    X_test = test_df[ALL_BASE_FEATURES]
    q_test_ens = predict_q_ensemble(s_ensemble_full, X_test)
    pi_test = build_policy_eps_greedy(q_test_ens, eps_best)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "p_mens_email": pi_test[:, ARM_TO_IDX["Mens E-Mail"]],
        "p_womens_email": pi_test[:, ARM_TO_IDX["Womens E-Mail"]],
        "p_no_email": pi_test[:, ARM_TO_IDX["No E-Mail"]],
    })

    # 8. checks & save
    assert submission["id"].nunique() == len(submission), "Duplicate ids in submission!"
    probs = submission[["p_mens_email", "p_womens_email", "p_no_email"]].values
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Row probabilities must sum to 1 in each row.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Found NaN or inf in probabilities.")

    return submission
    #submission.to_csv(submission_path, index=False)
    #print(f"\nSaved submission to: {submission_path}")




def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(train_and_predict(
        train_path="data/train.csv",
        test_path="data/test.csv",
        submission_path="submission.csv",
        random_state=42
    ))
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
