import functools
import json
import os
from multiprocessing import Pool

import pandas as pd
import pke
from tqdm import tqdm


# Function to split long text into multiple rows
def split_long_text(df, text_column, max_length=1000000):
    new_rows = []

    for index, row in df.iterrows():
        text = row[text_column]

        # Split text into chunks of max_length
        for i in range(0, len(text), max_length):
            new_row = row.copy()  # Copy other columns
            new_row[text_column] = text[i : i + max_length]  # Assign chunk
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def add_yake_keyword_column(text):
    extractor = (
        pke.unsupervised.YAKE()
    )  # Initialize inside worker to avoid pickling issues
    extractor.load_document(input=text, language="en")
    extractor.candidate_selection(n=1)
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=len(extractor.weights))
    keyphrases = [
        (str(keyphrase).strip().upper(), float(score))
        for keyphrase, score in keyphrases
    ]
    return keyphrases


# Function to apply multiprocessing with progress bar
def apply_multiprocessing_with_progress(df, func, n_processes, text_column):
    with Pool(n_processes) as pool:
        result = list(
            tqdm(
                pool.imap(func, df[text_column]),
                total=len(df),
                desc="Extracting Keywords",
            ),
        )
    return result


def gen_YAKE_keyword(path, text_column="text", n_rows=None):
    data_df = pd.read_parquet(path)

    if n_rows:
        data_df = data_df.head(n_rows)

    # Since in spacy we can't process more than 1M characters at a time, we split long text into multiple
    data_df = split_long_text(data_df, text_column, max_length=1000000)

    # Apply multiprocessing
    data_df["yake"] = apply_multiprocessing_with_progress(
        data_df,
        add_yake_keyword_column,
        int(0.8 * os.cpu_count()),
        text_column,
    )
    data_df["yake"] = data_df["yake"].apply(json.dumps)

    if n_rows:
        output_file_name = path.replace(".parquet", f"_yake_{n_rows}.parquet")
    else:
        output_file_name = path.replace(".parquet", "_yake.parquet")

    data_df.to_parquet(output_file_name, index=False)

    return {
        "path": "parquet",
        "data_files": output_file_name,
    }


def generate_keyword(path, method, text_column="text", n_rows=None):

    if method.lower() == "yake":
        return gen_YAKE_keyword(path, text_column, n_rows)
    else:
        raise ValueError("Invalid method")


if __name__ == "__main__":
    n_rows = 1000
    path = "/rhome/sawale/indus_traning/mlm-fine-tuning/mlm/data/cleaned_prod_dump_filtered_word_count.parquet"
    df = gen_YAKE_keyword(path, "text", n_rows)
