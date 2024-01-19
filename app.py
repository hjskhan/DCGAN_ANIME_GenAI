from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import base64
import torch
from PIL import Image
import io
import torch
import torch.nn as nn
import ignite.distributed as idist
from torchvision import utils as vutils
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

#Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
         
         # Input: 1x1x100, Output: 4x4x512
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Input: 4x4x512, Output: 8x8x256
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Input: 8x8x256, Output: 16x16x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Input: 16x16x128, Output: 32x32x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Input: 32x32x64, Output: 64x64x3
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Tanh activation function to map values to the image pixel range (e.g., [-1, 1]).
        )

    def forward(self, input):
        return self.main(input)
ngpu = 1
# Load your pre-trained DCGAN model
loaded_netG = Generator(ngpu=1)
state_dict = torch.load('generator_50_epoch.pth', map_location='cpu')

# Check if the keys have the 'module' prefix and remove it
if "module" in list(state_dict.keys())[0]:
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

loaded_netG.load_state_dict(state_dict)
loaded_netG.eval()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
latent_vector = 100
    
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "generated_image": None})

@app.post("/generate/")
async def generate_image(request: Request):
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    
    # Generate images using the loaded generator
    with torch.no_grad():
        # Assuming fixed_noise is the same tensor you used during training
        fake_images = loaded_netG(fixed_noise).detach().cpu()

    # Convert the PyTorch Tensor to an image
    pil_image = vutils.make_grid(fake_images, padding=2, normalize=True)
    pil_image = Image.fromarray((pil_image * 255).numpy().astype(np.uint8).transpose((1, 2, 0)))

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Encode the image bytes as base64 to embed in HTML
    encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

    return templates.TemplateResponse("index.html", {"request": request, "generated_image": encoded_image})