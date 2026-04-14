# data_preprocessing.py (最终版)

import os
import json  # 👈 导入json库
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42

def process_date_time(df, date_col='Date', time_col='Time'):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['Month'] = df[date_col].dt.month

    def time_period(t):
        if pd.isna(t): return np.nan
        try:
            hour = int(str(t).split(':')[0])
        except:
            return np.nan
        if 0 <= hour < 6:
            return 'late_night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 14:
            return 'noon'
        elif 14 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'

    df['Time_Period'] = df[time_col].apply(time_period)
    time_map = {'late_night': 1, 'morning': 2, 'noon': 3, 'afternoon': 4, 'evening': 5}
    df['Time_Period'] = df['Time_Period'].map(time_map)
    df.drop(columns=[date_col, time_col], inplace=True)
    return df


def clean_unknown(df):
    df.replace(['unknown', -1, '-1'], np.nan, inplace=True)
    return df


def lightgbm_impute(X_train, X_val, X_test):
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()
    imputation_models = {}

    for col in X_train.columns:
        if X_train[col].isna().sum() > 0:
            print(f"为列 '{col}' 训练插补模型...")
            not_null_train = X_train[X_train[col].notna()]
            X_fit = not_null_train.drop(columns=[col])
            y_fit = not_null_train[col]
            model_features = X_fit.columns.tolist()

            if y_fit.nunique() < 2:
                model = y_fit.mode()[0]
            else:
                if y_fit.nunique() < 20:
                    model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
                else:
                    model = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)

                cat_features_impute = X_fit.select_dtypes(include=['object', 'category']).columns.tolist()
                for cat_col in cat_features_impute:
                    X_fit[cat_col] = X_fit[cat_col].astype('category')

                model.fit(X_fit, y_fit, categorical_feature=cat_features_impute)

            imputation_models[col] = (model, model_features)

    print("\n开始插补所有数据集...")
    for df_impute in [X_train_imputed, X_val_imputed, X_test_imputed]:
        for col, (model, model_features) in imputation_models.items():
            if df_impute[col].isna().sum() > 0:
                is_null_mask = df_impute[col].isna()
                X_pred_df = df_impute.loc[is_null_mask, model_features]

                cat_features_impute = X_pred_df.select_dtypes(include=['object', 'category']).columns.tolist()
                for cat_col in cat_features_impute:
                    X_pred_df[cat_col] = X_pred_df[cat_col].astype('category')

                if not hasattr(model, 'predict'):
                    predictions = model
                else:
                    predictions = model.predict(X_pred_df)

                df_impute.loc[is_null_mask, col] = predictions

    print("插补完成。")
    return X_train_imputed, X_val_imputed, X_test_imputed


def undersample(X, y, k=2, random_state=RANDOM_STATE):
    if isinstance(y, pd.Series): y = y.values

    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    n_fatal = class_counts.get(1, 0)
    n_serious = class_counts.get(2, 0)
    n_slight = class_counts.get(3, 0)

    target_slight = int(k * (n_fatal + n_serious))
    target_slight = min(target_slight, n_slight)

    sampling_strategy = {1: n_fatal, 2: n_serious, 3: target_slight}

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    print("欠采样后:", dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


# ===============================
# 主流程 (最终版)
# ===============================
def main(input_path, output_dir, target_col):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    df = df.drop(columns=['Accident_Index'])

    # ======== 新增：移除经纬度特征 ========
    cols_to_drop = []
    if 'Latitude' in df.columns:
        cols_to_drop.append('Latitude')
    if 'Longitude' in df.columns:
        cols_to_drop.append('Longitude')
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"✅ 已移除地理坐标特征: {cols_to_drop}")
    # ====================================

    df = process_date_time(df)
    df = clean_unknown(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 / 3, random_state=RANDOM_STATE, stratify=y_temp)

    X_train, X_val, X_test = lightgbm_impute(X_train, X_val, X_test)

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

    X_train_encoded = preprocessor.transform(X_train)
    X_val_encoded = preprocessor.transform(X_val)
    X_test_encoded = preprocessor.transform(X_test)
    print("数据编码完成，已输出为NumPy数组。")

    print("\n欠采样前 (训练集):", y_train.value_counts().to_dict())
    X_train_resampled = X_train_encoded
    y_train_resampled = y_train.to_numpy()

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_resampled)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_encoded)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_encoded)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_resampled)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val.to_numpy())
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.to_numpy())

    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names.tolist(), f)
    print("✅ 特征名称已保存到 feature_names.json")

    print("\n✅ 数据预处理已正确完成。")
    print(f"Train: {X_train_resampled.shape}, Val: {X_val_encoded.shape}, Test: {X_test_encoded.shape}")


# CLI 接口 (无变动)
if __name__ == '__main__':
    input_path = r"E:\jiedan\交通事故\data\Accidents0514_copy.csv"
    output_dir = r"E:\jiedan\交通事故\data\output"
    target_col = "Accident_Severity"
    main(input_path, output_dir, target_col)
