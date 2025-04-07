import re

import pandas as pd
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
