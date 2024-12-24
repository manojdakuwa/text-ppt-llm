import datetime
from pptx import Presentation
from pptx.util import Inches, Pt
# from diffusers import StableDiffusionPipeline
# from transformers import DalleBartProcessor, DalleBartForConditionalGeneration
import torch
from PIL import Image
from io import BytesIO
import requests

# Load the pre-trained DALL-E Mini model and processor 
# model = DalleBartForConditionalGeneration.from_pretrained("flax-community/dalle-mini") 
# processor = DalleBartProcessor.from_pretrained("flax-community/dalle-mini")
# cache_dir = 'pytorch/stable_diffusion_model_cache'
def generate_image(text): 
    # if torch.cuda.is_available(): 
    #     device = "cuda" 
    #     print("CUDA is available. Using GPU.") 
    # else: 
    #     device = "cpu" 
    #     print("CUDA is not available. Using CPU.")
    # inputs = processor([text], return_tensors="pt") 
    # outputs = model.generate(**inputs, num_return_sequences=1) 
    # generated_images = processor.decode(outputs[0], skip_special_tokens=True) 
    # # Convert the generated image to a PIL image 
    # image = Image.open(BytesIO(requests.get(generated_images[0]).content))
    # Craiyon API endpoint 
    # api_url = "https://api.craiyon.com/generate" # Send the request to the Craiyon API 
    print("text for generating images: " + text)
    api_url = "https://bf.dallemini.ai/generate"
    response = requests.post(api_url, json={"prompt": text}) 
    response.raise_for_status()
    # Retrieve the generated image 
    image_url = response.json()["images"][0] 
    image_response = requests.get(image_url) 
    image = Image.open(BytesIO(image_response.content))
    image_stream = BytesIO() 
    image.save(image_stream, format='PNG') 
    image_stream.seek(0)
    return image_response

def create_ppt(slides_data, file_name='output_presentation.pptx'):
    print("slides_data: ", slides_data)
    prs = Presentation()
    try:
        for slide_content in slides_data:
            slide_layout = prs.slide_layouts[1]  # Title and Content layout
            slide = prs.slides.add_slide(slide_layout)
            title = slide.shapes.title
            title.text_frame.paragraphs[0].font.size = Pt(2)
            content = slide.placeholders[1]
            title.text = slide_content['title'] 
            content.text = slide_content['text']
            for paragraph in content.text_frame.paragraphs: 
                paragraph.font.size = Pt(14)
            # image = generate_image(slide_content['title'])  # Replace with paid actual image generation function if needed

            # slide.shapes.add_picture("",Inches(1), Inches(2), width=Inches(6))

        file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_presentation.pptx")
        prs.save(file_name)
        return "200"
    except Exception as e:
        print("Error creating presentation", str(e))
        return "404"

def create_slide_data(response):
    try:
        return [ {'title': f'{item["title"]}', 'text': item["summarize_text"]} for i, item in enumerate(response)]
    except Exception as e:
        print("Error creating slide")

def generate_ppt_csv(grouped_data, charts): 
    prs = Presentation()
    try:
        for context, group in grouped_data.groupby('cluster'): 
            print("context: ", context)
            print("group; ", group)
            slide = prs.slides.add_slide(prs.slide_layouts[5]) 
            title = slide.shapes.title 
            title.text = f"Context {str(context).capitalize()}" # Summary text 
            title.text_frame.paragraphs[0].font.size = Pt(14)
            summary_text = ' '.join(group['product_details'].tolist()) 
            textbox = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(8.5), Inches(2)) 
            tf = textbox.text_frame 
            tf.text = summary_text # Add chart image if exists 
            p = tf.add_paragraph() 
            p.text = tf.text 
            p.font.size = Pt(12)
            if context in charts: 
                slide.shapes.add_picture(charts[context], Inches(1), Inches(3.5), Inches(8.5), Inches(3.5)) 
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_presentation.pptx") 
            prs.save(filename)
        return "200"
    except Exception as e:
        print("Error generating presentation", str(e))
        return "404"

