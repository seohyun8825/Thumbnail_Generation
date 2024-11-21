
from model.Modified_xl import AddedUserLatentPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from model.mllm import local_llm,GPT4
import torch
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('API_KEY')


pipe = AddedUserLatentPipeline.from_pretrained("comin/IterComp",torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
pipe.enable_xformers_memory_efficient_attention()
## User input
prompt= 'Four snapshots of a dog'
para_dict = GPT4(prompt,key=api_key)
split_ratio = para_dict['Final split ratio']
regional_prompt = para_dict['Regional Prompt']
negative_prompt = ""

image_path = [
    "user_img/a.png",
    "user_img/b.png",
    "user_img/happy_sitting.png",
]

print("[DEBUG] Preparing to call the pipeline...")
images = pipe(
    prompt = regional_prompt,
    split_ratio = split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
    batch_size = 1, #batch size
    base_ratio = 0.5, # The ratio of the base prompt    
    base_prompt= prompt,       
    num_inference_steps=20, # sampling step
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    width = 1024, 
    seed = 2468,# random seed
    guidance_scale = 7.0,
    user_images = image_path,
).images[0]
print("[DEBUG] Pipeline call finished.")
images.save("test.png")
