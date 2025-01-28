import re

import dask.bag as db
import pandas as pd
from dask.diagnostics import ProgressBar

# Path to the file
file_path = "../data/sde_init_check.csv"
output_file = "output.parquet"

# Read the text file into a Dask bag
bag = db.read_text(file_path, linedelimiter="\u202E\u2064\u202D")

# Define a function to parse each line
def parse_line(line):
    parts = line.split(";")

    # Ensure there are enough fields, otherwise fill with empty strings
    url = parts[0] if len(parts) > 0 else ""
    first_field = parts[1] if len(parts) > 1 else ""
    second_field = parts[2] if len(parts) > 2 else ""
    third_field = parts[3] if len(parts) > 3 else ""
    fourth_field = parts[4] if len(parts) > 4 else ""
    text = ""

    # filter url
    url_match = re.search(r"https?://[^\s;]+", url)
    url = url_match.group(0) if url_match else ""

    # filter text from
    fourth_field = fourth_field.split("\n", 1)
    if len(fourth_field) == 2:
        fourth_field, text = fourth_field

    return (url, first_field, second_field, third_field, fourth_field, text)


# Map the parse_line function over the Dask bag
parsed_bag = bag.map(parse_line)

# Convert the result into a Pandas DataFrame
with ProgressBar():
    parsed_bag.to_dataframe(
        meta={
            "url": "object",
            "first_field": "object",
            "second_field": "object",
            "third_field": "object",
            "fourth_field": "object",
            "text": "object",
        },
    ).to_parquet(output_file)
