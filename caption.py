from transformers import VisionEncoderDecoderModel,ViTImageProcessor,AutoTokenizer
import torch
from PIL import Image

model=VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor=ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer=AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length=16
num_beams=4
gen_kwargs={"max_length":max_length,"num_beams":num_beams}

def predict_caption(image_path):
    images = []
    '''for image_path in image_path:
        i_image=Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)'''
        
    pixel_values = feature_extractor(
        images=[image_path],return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values,**gen_kwargs)
    
    preds = tokenizer.batch_decode(output_ids,skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    pred=''.join(preds)
    print("Final caption is: ".join(preds))
    return pred
image=Image.open('horse.jpg')
print(predict_caption(image_path=image))
    
