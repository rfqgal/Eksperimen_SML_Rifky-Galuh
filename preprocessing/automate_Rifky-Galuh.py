import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    categorical_features = df.select_dtypes(include='object').columns.to_list()
    numerical_features = ['Sex', 'Age', 'Settlement size']

    # 1. Drop Unused Feature
    df = df.drop(columns=['ID'])

    # 2. Handle Outliers
    Q1 = df[numerical_features + ['Income']].quantile(0.25)
    Q3 = df[numerical_features + ['Income']].quantile(0.75)
    IQR = Q3 - Q1
    filter_outliers = ~((df[numerical_features + ['Income']] < (Q1 - 1.5 * IQR)) |
                        (df[numerical_features + ['Income']] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[filter_outliers]

    # 3. Data Splitting
    X = df.drop(columns=['Income'])
    y = df['Income']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=123)

    # 4. Encoding & Scaling
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_enc.toarray() if hasattr(X_train_enc, "toarray") else X_train_enc,
                              columns=feature_names, index=y_train.index)
    X_test_df = pd.DataFrame(X_test_enc.toarray() if hasattr(X_test_enc, "toarray") else X_test_enc,
                             columns=feature_names, index=y_test.index)

    train_final = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test_final = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)
    return train_final, test_final

# csv_path = f'../sgdata.csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'sgdata_raw.csv')

train_final, test_final = preprocessing_pipeline(csv_path)
train_final.to_csv("sgtrain.csv", index=False)
test_final.to_csv("sgtest.csv", index=False)
