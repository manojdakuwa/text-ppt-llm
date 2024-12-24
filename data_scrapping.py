from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer 
import numpy as np
# from langchain_community.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import os
# from dotenv import load_dotenv
# load_dotenv()

# os.environ['OPEN_API_KEY'] = os.environ("OPEN_API_KEY")

# Load the model and tokenizer
model_name = "t5-base" #'facebook/bart-large-cnn'  # Use a known summarization model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt", device=-1)

# # Function to generate titles and summaries using OpenAI's GPT-3 for text to image generation
# def generate_title_and_summary(chunk): 
#     openai.api_key = os.environ['OPEN_API_KEY'] # Replace with your OpenAI API key 
#     prompt = f"Generate a title and summary for the following text:\n\nText: {chunk}\n\nTitle:\n\nSummary:" 
#     response = openai.Completion.create( engine="text-davinci-003", # Change to the appropriate model name if needed 
#                                         prompt=prompt, max_tokens=150, n=1, stop=None, temperature=0.7, ) 
#     text = response.choices[0].text.strip() # Extract Title and Summary from the response 
#     title = text.split("\n\nTitle: ")[1].split("\n\nSummary: ")[0].strip() 
#     summary = text.split("\n\nSummary: ")[1].strip() 
#     return title, summary

# # Function to generate titles and summaries using LangChain 
# def generate_title_and_summary(chunk): 
#     openai_llm = OpenAI(api_key=os.environ['OPEN_API_KEY'], model_name="gpt-3.5-turbo-instruct") # Replace with your OpenAI API key 
#     prompt_template = PromptTemplate( input_variables=["text"], template="Generate a title and summary for the following text:\n\nText: {text}\n\nTitle: [Insert Title]\n\nSummary: [Insert Summary]" ) 
#     llm_chain = LLMChain(llm=openai_llm, prompt_template=prompt_template) 
#     response = llm_chain.run({"text": chunk}) # Extract Title and Summary from response 
#     title = response.split("\n\nTitle: ")[1].split("\n\nSummary: ")[0].strip() 
#     summary = response.split("\n\nSummary: ")[1].strip() 
#     return title, summary

# Function to generate titles and summaries using transformers library 

def generate_title(text, top_n=5): # you can also use open ai for better result and better performance commented one above
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        
        # Extract top keywords
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names_out()
        top_keywords = [features[i] for i in indices[:top_n]]
        
        # Join top keywords to form a title
        title = ' '.join(top_keywords)
        return title.capitalize()
    except Exception as e:
        print("Error generating title:", str(e))
        return None

def generate_title_and_summary(chunk): 
    title_prompt = f"Generate a concise and informative title for the following text:\n\n{chunk}" 
    summary_prompt = f"Summarize the following text:\n\n{chunk}" 
    title = generate_title(chunk) #summarizer(title_prompt, max_length=40, min_length=5, do_sample=False)[0]['summary_text'] 
    summary = summarizer(summary_prompt, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] 
    return title, summary

def text_splitter(text):
    text_splitter_= RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=128, separators=["\n\n", "\n",  ".", "!", "?", ",", " ", ""]
    )
    texts = text_splitter_.split_text(text)
    return texts
# Summarize text
# def summarize_text(text):
#     # text_splitter = RecursiveCharacterTextSplitter(
#     #     chunk_size=1024, chunk_overlap=128, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#     # )
#     # texts = text_splitter.split_text(text)
#     # summaries = []
#     # for chunk in texts:
#     summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
#     # summaries.append(summary[0]['summary_text'])
#     return " ".join(summary[0]['summary_text'])

# # Generate title
# def generate_title(text):
#     prompt = f"Generate a concise and informative title for the following text:\n{text}"
#     title_summary = summarizer(prompt, max_length=30, min_length=5, do_sample=False)
#     return title_summary[0]['summary_text']

# Main function
def url_scrapping_query(url=None, text=None):
    if url:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            content = " ".join(doc.page_content for doc in documents)
            texts = text_splitter(content)
            # print("texts: ", texts)
            results = [] 
            for chunk in texts: 
                title, summary = generate_title_and_summary(chunk)
                results.append({"title": title, "summarize_text": summary})
            return results
        except Exception as e:
            print(str(e))
            return []
        
    if text:
        # print("text: " + text)
        try:
            texts = text_splitter(text)
            results = [] 
            for chunk in texts: 
                title, summary = generate_title_and_summary(chunk) 
                results.append({"title": title, "summarize_text": summary}) 
            return results
        except Exception as e:
            print("Exception", e)
            return []

# # Example usage
# url = ''  # Replace with your URL
# result = main(url)
# print(result)
