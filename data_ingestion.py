import pandas as pd
import json
import io

def load_csv(file_content):
    return pd.read_csv(io.BytesIO(file_content))

def load_json(file_content):
    return json.load(io.BytesIO(file_content))

def ingest_data(file_content, file_type):
    if file_type == 'csv':
        return load_csv(file_content)
    elif file_type == 'json':
        return load_json(file_content)
    else:
        raise ValueError("Unsupported file type")
