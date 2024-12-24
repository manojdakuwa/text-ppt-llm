from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from data_ingestion import ingest_data
from data_processing import process_contexts #, summarize_text, process_numerical_data, generate_content
from chart_generation import create_charts
from ppt_generation import create_ppt, create_slide_data, generate_ppt_csv
from data_scrapping import url_scrapping_query, generate_title_and_summary
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# # Load the summarization model
# model_name = 'facebook/bart-large-cnn'#'t5-small'  # Change this to any other model if needed

# Use a different summarization model 
model_name = 'facebook/bart-large-cnn' # A lighter model that works well for summarization # Load model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_name, framework="pt", device=-1)

@app.post("/process")
async def process_data(
    files: Optional[List[UploadFile]] = None,
    text: Optional[str] = Form(None)
):
    response = []
    if files:
        for file in files:
            file_type = file.filename.split('.')[-1]

            if file_type == 'csv':
                '''
                The CSv file will conatin following columns "product_details", "year", "price"
                '''
                content = await file.read()
                data = ingest_data(content, file_type)
                grouped_data = process_contexts(data) 
                charts = create_charts(grouped_data) 
                res = generate_ppt_csv(grouped_data, charts)
            else:
                '''
                [{'Topics':""},
                {'Topics':""}]
                This is object Array have topic key
                '''
                for key in data.keys(): 
                    data=response.copy()
                    data["title"], data["summary"] = generate_title_and_summary(data[key])
                    # data["title"] = generate_title(data[key])
                    response.append(data)
                
                slides_data = create_slide_data(response)
                res = create_ppt(slides_data, charts=charts, file_name='output_presentation.pptx')

    if text:
        # Handle URLs
        if text.startswith("http://") or text.startswith("https://"):
            text= url_scrapping_query(text, None)
            # response = process_text(text)
            # response = text
            slides_data = create_slide_data(text)
            print("slides_data: ", slides_data)
            # charts = create_charts(slides_data) if slides_data else {}
            res = create_ppt(slides_data, file_name='output_presentation.pptx')
        else:
            #print(text, len(text))
            text= url_scrapping_query(None, text)
            # response = text
            slides_data = create_slide_data(text)
            # charts= create_charts(slides_data) if slides_data else {}
            res = create_ppt(slides_data, file_name='output_presentation.pptx')
            #generated_content = generate_content(processed_text, content_generator)
            #processed_text = generated_content
        
        # summary = summarize_text(processed_text, summarizer)
        # print("summary: ", summary)
        # response['text_summary'] = summary

    # slides_data = create_slide_data(response)
    # charts = ['chart.png'] if 'chart' in response else []
    # Generate PowerPoint
    # create_ppt(slides_data, charts=charts, file_name='output_presentation.pptx')
    if res == "200":
        return JSONResponse({'message': 'Processing completed. Check output_presentation.pptx for results.'})
    else:
        return JSONResponse({'message': 'Error occurred during processing.'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
