from data_ingestion import ingest_data
from data_processing import process_text, summarize_text, process_numerical_data
from chart_generation import create_chart
from ppt_generation import create_ppt

# Example input paths and types
file_path = 'data.csv'
file_type = 'csv'
llm_model = 't5-small'  # or any other model like 'gpt-3', 'bert'

# Ingest data
data = ingest_data(file_path, file_type)

# Process data
if file_type == 'json' or file_type == 'api':
    for key in data.keys():
        data[key] = process_text(data[key])
        data[key] = summarize_text(data[key], llm_model)

if file_type == 'csv':
    numerical_data = process_numerical_data(data)
    create_chart(numerical_data, chart_type='bar', file_name='chart.png')

# Prepare data for PowerPoint slides
slides_data = [
    {'title': 'Slide 1 Title', 'text': data[key] if file_type != 'csv' else 'Numerical Data Chart'}
    for key in data.keys()
]

charts = ['chart.png'] if file_type == 'csv' else []

# Generate PowerPoint
create_ppt(slides_data, charts=charts)
