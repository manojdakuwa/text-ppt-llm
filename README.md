# GenAI-based Tool for Automated Visual Representations

This project is a generative AI-based tool capable of transforming structured and unstructured outputs (e.g., text, tables, and numbers) into polished, visually appealing representations in Microsoft PowerPoint and Word formats. The tool leverages large language models (LLMs) for processing inputs and generating outputs.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Modules](#modules)
  - [Data Ingestion](#data-ingestion)
  - [Data Processing and Summarization](#data-processing-and-summarization)
  - [Chart Generation](#chart-generation)
  - [PowerPoint Generation](#powerpoint-generation)
- [Next Steps](#next-steps)
- [License](#license)

## Project Structure

gen_ai_tool/ 
├── app.py 
├── chart_generation.py 
├── data_ingestion.py 
├── data_processing.py 
├── data_scrapping.py 
├── main.py 
├── ppt_generation.py 
└── requirements.txt 
├── static/ 
│ ├── index.html 
│ └── app.js 
└── README.md


## Requirements

- pandas
- matplotlib
- python-pptx
- transformers 
- requests
- uvicorn
- FastAPI
- tensorflow
- python-multipart
- sentencepiece
- bs4
- textblob
- langchain
- langchain_community
- langchain_text_splitters
- diffusers
- accelerate
- scikit-learn
- tf-keras
- numpy
- dalle-mini
- pillow


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/gen_ai_tool.git
    cd gen_ai_tool
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. a. Start the FastAPI server:
    ```bash
    uvicorn app:app --reload
    ```
   b. Start the frontend running locally

2. Use a tool like Postman or Curl to send a POST request to `http://127.0.0.1:8000/process` with your files and/or text.

## Features

- Accepts structured (e.g., tables, numbers) and unstructured (e.g., text, paragraphs) data as input via API.
- Generates PowerPoint presentations with:
  - Slide layouts, headers, and bullet points.
  - Charts or graphs for numerical data.
  - Customizable color schemes and templates.
- Intelligent grouping of content into slides.
- Context-based summarization for verbose inputs.
- Automatic chart generation from numerical data.

## Modules

### API Creation

Uses FastAPI to create an API for handling multiple file types and text inputs.

### Data Ingestion

Handles inputs from JSON, CSV files, and API URLs.

### Data Processing and Summarization

Uses a large language model (e.g., GPT, T5, BERT) to process and summarize text data.

### Chart Generation

Creates charts from numerical data.

### PowerPoint Generation

Generates the final PowerPoint presentations.

## Next Steps

1. **Enhance Data Processing and Summarization**: Improve processing and summarization logic for complex inputs.
2. **Expand Chart Types**: Add support for more chart types (e.g., pie charts, scatter plots).
3. **Create Word Document Generation**: Implement similar functionality for Word document generation.
4. **Modular Expansion**: Ensure the code is modular for easy expansion and maintenance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
