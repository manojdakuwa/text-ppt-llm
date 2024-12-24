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
- [Design & Architecture](#design-and-architecture)
  - [Design & Architecture](#design-and-architecture)
  - [Model Loading and Caching](#model-loading-and-caching)
  - [Image Generation](#image-generation)
  - [PowerPoint Integration](#powerpoint-integration])
  - [Error Handling](#error-handling)
- [Key Challenges Faced and How They Were Addressed](#key-challenges-faced-and-how-they-)
  - [Model Download and Caching](#model-download-and-caching)
  - [Error Handling](#error-handling)
  - [Integration with PowerPoint](#integration-with-)
- [Possible Enhancements for Future Iterations](#possible-enhancements-for-future-iterations)
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

## Design & Architecture

Design and Architecture
This project leverages the Hugging Face transformers library and diffusers library for text-to-image generation. The key components include:

### Model Loading and Caching 
We use the StableDiffusionPipeline from the diffusers library to load and cache the Stable Diffusion model locally. This ensures efficient image generation without the need for repeated downloads.

### Image Generation 
The pipeline processes a text prompt to generate images, which are then converted to a Base64 string for easy handling.

### PowerPoint Integration 
Using the python-pptx library, the generated images are inserted into PowerPoint presentations directly from the Base64 string, avoiding the need to save images locally.

### Error Handling 
Robust error handling ensures that any issues during model loading, image generation, or PowerPoint creation are gracefully managed, with informative logging to aid troubleshooting.

## Key Challenges Faced and How They Were Addressed

### Model Download and Caching

Challenge: Slow download times and large model sizes.

Solution: Implemented local caching using the huggingface_hub library's snapshot functionality. This ensures that models are downloaded once and reused without requiring repeated downloads. In my case it was taking more than 3 hours to download, could be beacuase of internet connectivity issues or other. So if you have very good system, I will suggest yo download locally. else you can create api key and use them in prject but make sure you take care of token. it can cause money.

### Error Handling

Challenge: Handling various exceptions, such as network errors during model download and issues with image generation.

Solution: Added try-except blocks to capture and log errors, ensuring that the application fails gracefully and provides useful information for debugging.

### Integration with PowerPoint

Challenge: Inserting images into PowerPoint without saving them locally.

Solution: Converted generated images to Base64 strings and used python-pptx to insert these images directly into the presentation.

## Possible Enhancements for Future Iterations

1. **Model Optimization**:

Objective: Reduce model size and improve load times.

Approach: Explore quantization techniques and model distillation to create smaller, faster versions of the models. Model Evaulation is important for performance.

Enhanced Error Handling:

2. **Objective**: Improve robustness and user experience.

Approach: Implement more detailed logging and user-friendly error messages. Add retries for network-related errors.

User Interface:

3. **Objective**: Provide a more intuitive and interactive experience.

Approach: Develop a web-based UI using frameworks like Flask or Streamlit to allow users to input text prompts and generate images without writing code.

4. **Additional Model Support**:

Objective: Broaden the range of text-to-image models supported.

5. **Approach**: Integrate additional models from Hugging Face and other repositories, providing users with more options for image generation.

Cloud Integration:

Objective: Leverage cloud resources for large-scale image generation.

6. **Approach**: Implement cloud-based processing using services like AWS or Google Cloud to handle larger workloads and improve performance.

## Next Steps

1. **Enhance Data Processing and Summarization**: Improve processing and summarization logic for complex inputs.
2. **Expand Chart Types**: Add support for more chart types (e.g., pie charts, scatter plots).
3. **Create Word Document Generation**: Implement similar functionality for Word document generation.
4. **Modular Expansion**: Ensure the code is modular for easy expansion and maintenance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
