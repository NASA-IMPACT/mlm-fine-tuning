import pandas as pd
# import dask.dataframe as dd
from urllib.parse import urlparse, urlunparse
# from urllib.parse import quote, unquote

def normalize_url(url):
    parsed = urlparse(url)
    # Remove 'www.' if it exists in the netloc
    netloc = parsed.netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]  # Strip the 'www.' prefix

    normalized = urlunparse(parsed._replace(
        # scheme=parsed.scheme.lower(),
        scheme="http",
        netloc=netloc,
        path=parsed.path.strip('/')
    ))
    return normalized


df = pd.read_parquet("../data/parsed_sde_init_check")
urls_df = pd.read_json('../data/candidate_urls.zip')

df = df.drop_duplicates(subset=['url'])
urls_df = urls_df.drop_duplicates(subset=['url'])


print(f"Inital Shape of parsed_sde_init_check: {df.shape}")
print(f"Inital Shape of candidate_urls: {urls_df.shape}")

print(f'Unique urls in parsed_sde_init_check: {df["url"].nunique()}')
print(f'Unique urls in candidate_urls: {urls_df["url"].nunique()}')

print("In parsed_sde_init_check, urls strarts with:")
print(df["url"].apply(lambda x: x.split("/")[0]).value_counts())
print("In candidate_urls, urls strarts with:")
print(urls_df["url"].apply(lambda x: x.split("/")[0]).value_counts())

df['normalized_url'] = df['url'].apply(normalize_url)
urls_df['normalized_url'] = urls_df['url'].apply(normalize_url)

result = df.merge(urls_df, on='normalized_url', how='inner')

print(f"Shape of parsed_sde_init_check: {df.shape}")
print(f"Shape of candidate_urls: {urls_df.shape}")
print(f"Shape of result: {result.shape}")

