import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_raw_json_training_file_mc_ml(data):
    df = pd.DataFrame(data)

    def clean_labels(label_list):
        if not label_list:
            return ["non-TDAMM"]
        return [label.strip() for label in label_list]

    def preprocess_text(text):
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    df["labels"] = df["labels"].apply(clean_labels)
    df["full_text"] = df["full_text"].apply(preprocess_text)

    mlb = MultiLabelBinarizer()
    labels_binarized = mlb.fit_transform(df["labels"])
    labels_df = pd.DataFrame(labels_binarized, columns=mlb.classes_)
    df = pd.concat([df, labels_df], axis=1)
    df.drop("labels", axis=1, inplace=True)

    return df


def tdamm_preprocess(df, data_config, seed, nrows=None):
    print(data_config.get("path"))
    df = preprocess_raw_json_training_file_mc_ml(df)
    if nrows:
        df = df.head(nrows)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    input_columns = [data_config.get("input_text_col")]
    non_target_columns = data_config.get("non_target_cols")
    target_columns = [c for c in df.columns if c not in non_target_columns]

    train_x = train_df[input_columns]
    train_y = train_df[target_columns].astype(float)
    test_x = test_df[input_columns]
    test_y = test_df[target_columns].astype(float)
    val_x = test_x.copy()
    val_y = test_y.copy()

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        {
            "input_columns": input_columns,
            "target_columns": target_columns,
            "non_target_columns": non_target_columns,
        },
    )
