from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

video_path = "./VideoHallucer/videohallucer_datasets/external_factual/videos/_GTwKEPmB-U_5183.mp4"

# CACHE_DIR="/data/harold/ViDPO/cache"

model_path = "./checkpoints/VideoLLaVA/VistaDPO" 
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, device=device, model_name=model_name)
inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

# our pipeline
frame_dir, _ = os.path.splitext(video_path)
decode2frame(video_path, frame_dir, verbose=True)
question="What is the evident theme in the video?"
response = inference_model.generate(
    question=question,
    modal_path=frame_dir,
    temperature=0,
)
print(response)

# using decord 
response = inference_model.generate(
    question=question,
    modal_path=video_path,
    temperature=0,
    video_decode_backend="decord",
)
print(response)