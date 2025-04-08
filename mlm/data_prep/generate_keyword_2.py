import gc
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pke
import psutil
from tqdm.auto import tqdm


# Calculate optimal resource usage based on system capabilities
def get_optimal_resources(reserve_memory_gb=50):
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = (
        total_memory_gb - reserve_memory_gb
    )  # Reserve some memory for OS and other processes
    cpu_count = os.cpu_count()

    print(f"System has {total_memory_gb:.1f} GB RAM and {cpu_count} logical cores")
    print(f"Using up to {available_memory_gb:.1f} GB RAM for processing")

    return {
        "available_memory_gb": available_memory_gb,
        "cpu_count": cpu_count,
    }


# Calculate optimal chunk size based on available memory
def calculate_chunk_size(df, available_memory_gb):
    # Estimate memory usage per row (in GB)
    sample_memory_usage = df.memory_usage(deep=True).sum() / (1024**3) / len(df)

    # Determine how many rows we can process at once with our available memory
    # Using 50% of available memory for the chunk to be safe
    optimal_chunk_size = int((available_memory_gb * 0.5) / sample_memory_usage)

    # Cap at a reasonable value to avoid too large chunks
    return min(optimal_chunk_size, 50000)


# Function to split long text into multiple rows - using vectorized operations where possible
def split_long_text(df, text_column, max_length=1000000):
    # Get lengths of all texts
    text_lengths = df[text_column].str.len()
    needs_splitting = text_lengths > max_length

    if not needs_splitting.any():
        return df

    # Process rows that don't need splitting
    result_df = df[~needs_splitting].copy()

    # Only process rows that need splitting
    split_rows = []
    split_df = df[needs_splitting]

    print(f"Splitting {needs_splitting.sum()} long text rows...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        text = row[text_column]
        chunk_count = (len(text) // max_length) + (
            1 if len(text) % max_length > 0 else 0
        )

        for i in range(chunk_count):
            new_row = row.copy()
            new_row[text_column] = text[i * max_length : (i + 1) * max_length]
            split_rows.append(new_row)

    # Combine results
    if split_rows:
        split_df = pd.DataFrame(split_rows)
        result_df = pd.concat([result_df, split_df], ignore_index=True)

    return result_df


def add_yake_keyword_column(text):
    """Extract ALL keywords from text using YAKE with n-gram=1 only"""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return []

    try:
        extractor = pke.unsupervised.YAKE()
        extractor.load_document(input=text, language="en")
        # Set n-gram to 1 as requested
        extractor.candidate_selection(n=1)
        extractor.candidate_weighting()

        # Get ALL available keyphrases - no limit
        keyphrases = extractor.get_n_best(n=len(extractor.weights))

        return [
            (str(keyphrase).strip().upper(), float(score))
            for keyphrase, score in keyphrases
        ]
    except Exception as e:
        print(f"Error processing text: {str(e)[:100]}...")
        return []


# Parallel processing with chunking and optimized resource usage
def parallel_process(df, func, text_column, n_processes, chunk_size):
    results = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        print(
            f"Processing chunk {i//chunk_size + 1}/{total_chunks} ({len(chunk)} rows)...",
        )

        with Pool(processes=n_processes) as pool:
            chunk_results = list(
                tqdm(
                    pool.imap(func, chunk[text_column]),
                    total=len(chunk),
                    desc=f"Chunk {i//chunk_size + 1}/{total_chunks}",
                ),
            )

        results.extend(chunk_results)

        # Force garbage collection after each chunk
        gc.collect()

    return results


def gen_YAKE_keyword(path, text_column="text", n_rows=None):
    # Get system resources
    resources = get_optimal_resources()

    # Determine optimal number of processes - use 80% of available cores
    n_processes = int(resources["cpu_count"] * 0.8)
    print(f"Using {n_processes} processes for parallel processing")

    # Read data efficiently - loading all columns at once
    print(f"Reading parquet file: {path}")
    if n_rows:
        data_df = pd.read_parquet(path).head(n_rows)
    else:
        data_df = pd.read_parquet(path)

    print(
        f"Loaded dataframe with {len(data_df)} rows and {len(data_df.columns)} columns",
    )

    # Split long texts
    data_df = split_long_text(data_df, text_column, max_length=1000000)
    print(f"After splitting, processing {len(data_df)} rows total")

    # Calculate optimal chunk size based on available memory
    chunk_size = calculate_chunk_size(data_df, resources["available_memory_gb"])
    print(f"Processing in chunks of {chunk_size} rows")

    # Process in parallel with chunking
    print("Extracting ALL keywords with n-gram=1 (unigrams only)...")
    data_df["yake"] = parallel_process(
        data_df,
        add_yake_keyword_column,
        text_column,
        n_processes,
        chunk_size,
    )

    # Sample of keyword count distribution
    keyword_counts = [len(kw) for kw in data_df["yake"].head(1000)]
    if keyword_counts:
        print(f"Keyword count statistics (sample of 1000 rows):")
        print(f"  Min: {min(keyword_counts) if keyword_counts else 0}")
        print(f"  Max: {max(keyword_counts) if keyword_counts else 0}")
        print(
            f"  Avg: {sum(keyword_counts)/len(keyword_counts) if keyword_counts else 0:.1f}",
        )

    # Convert to JSON strings
    print("Converting results to JSON...")
    data_df["yake"] = data_df["yake"].apply(lambda x: json.dumps(x) if x else "[]")

    # Save results
    if n_rows:
        output_file_name = path.replace(".parquet", f"_yake_unigram_{n_rows}.parquet")
    else:
        output_file_name = path.replace(".parquet", "_yake_unigram.parquet")

    print(f"Saving results to {output_file_name}")
    data_df.to_parquet(output_file_name, index=False, compression="snappy")

    return {
        "path": "parquet",
        "data_files": output_file_name,
    }


def generate_keyword(path, method, text_column="text", n_rows=None):
    if method.lower() == "yake":
        return gen_YAKE_keyword(path, text_column, n_rows)
    else:
        raise ValueError(f"Method '{method}' not supported")


if __name__ == "__main__":
    n_rows = None
    path = "/rhome/sawale/indus_traning/mlm-fine-tuning/mlm/data/cleaned_prod_dump_filtered_word_count.parquet"
    print(f"Starting keyword extraction for {path}")
    result = gen_YAKE_keyword(path, "text", n_rows)
    print("Complete!")
    print(result)
