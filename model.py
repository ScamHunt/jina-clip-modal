import modal

app = modal.App("jina-ai-clip-v1")


docker_image = modal.Image.debian_slim(python_version="3.11").run_commands(
   [ 
    "apt-get -y update; apt-get -y install curl",
    "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
    "apt-get install git-lfs -y",
    "git lfs install",
    "git clone https://huggingface.co/jinaai/jina-clip-v1 /weights",]
).pip_install(
  [  "timm==1.0.9",
    "transformers==4.44.2",
    "einops==0.8.0",
    "pillow==10.4.0"]

)


with docker_image.imports():
    from transformers import AutoModel
    from PIL import Image
    import requests
    from typing import Dict
    from io import BytesIO



app = modal.App(image=docker_image)

@app.cls(gpu="T4")
class JinaClipV1:
    @modal.enter()
    def load_model(self):
        #TODO  need to figure out why this is not working
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path="/weights", trust_remote_code=True)
        
    @modal.method()
    def predict(self,text:str, image):
        text_embeddings  =  self.model.encode_text([text])
        image_embeddings = self.model.encode_image([image])
        return text_embeddings.tolist(), image_embeddings.tolist()







@app.function(image=docker_image)
def download_img(image_url:str):
    print(image_url)
    image_bytes = requests.get(image_url)
    print(image_bytes)
    image = Image.open(BytesIO(image_bytes.content))
    return image


@app.function(image=docker_image)
@modal.web_endpoint(method="POST")
def embed(payload:dict) :
    print(payload)
    image = download_img.remote(payload["image_url"])
    text_embeddings, image_embeddings = JinaClipV1().predict.remote(payload["text"], image)
    return {"text_embeddings": text_embeddings, "image_embeddings": image_embeddings}
