from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob 
from sklearn.cluster import KMeans 
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_contexts(df, n_clusters=5):
    try:
        texts = df['product_details'].tolist() # Assume the column with text data is named 'text' 
        embeddings = extract_context(texts) 
        flattened_embeddings = [embedding[0] for embedding in np.array(embeddings)] # Flatten embeddings  
        clusters = cluster_contexts(flattened_embeddings, n_clusters) 
        # print("Clusters: ", clusters)
        df['cluster'] = clusters 
    except Exception as e:
        print("Error processing contexts", str(e))
    return df

def extract_context(texts): # Use a pre-trained transformer model for embedding 
    try:
        model_name = 'sentence-transformers/all-mpnet-base-v2' 
        nlp = pipeline("feature-extraction", model=model_name) 
        embeddings = nlp(texts)
        pooled_embeddings = [np.mean(embedding, axis=1).flatten() for embedding in embeddings] # Flatten embeddings return 
        # embedding_array = np.array(pooled_embeddings) 
        # Normalize embeddings 
        scaler = StandardScaler() 
        normalized_embeddings = scaler.fit_transform(pooled_embeddings)
        return normalized_embeddings
    except Exception as e:
        print("Error extracting context", str(e))

def cluster_contexts(embeddings, n_clusters): # Use KMeans clustering to group similar contexts
    try: 
        embeddings = np.array(embeddings)
        # print("embeddings: ", embeddings)
        if embeddings.ndim == 1: 
            embeddings = embeddings.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
        clusters = kmeans.fit_predict(embeddings) 
        # print("embeddings clusters: ", clusters)
        return clusters
    except Exception as e:
        print("Error clustering contexts", str(e))

# def summarize_text(text, summarizer):
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name) 
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer, framework="pt", device=-1)

#     # summarizer = pipeline("summarization", model=model)
#     summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
#     return summary[0]['summary_text']

# def generate_content(text, model): 
#     generated = model(text, max_length=150, min_length=30, do_sample=True) 
#     return generated[0]['generated_text']

# def process_numerical_data(data):
#     # Implement processing of numerical data if needed - not coded this yet, can we done in future versions.
#     return data
