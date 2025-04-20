# Stable Diffusion Image Generator (Google Colab)

This project uses the [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) model via Hugging Face's `diffusers` library to generate high-quality AI images based on your text prompts — **completely free on Google Colab**.

## 🚀 Features

- Generate unlimited images from text prompts using Stable Diffusion 2.1
- High resolution outputs (default: 1000x1000)
- GPU-accelerated with CUDA on Colab
- No API key or account required (public weights)
- Ready to use — just open in Colab and run

## 🔗 Google Colab

👉 **Open this notebook in Colab:**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fl7TMNjtq1h5hRaBgP324edyzRvoEOm2)

## 🧱 Requirements

The notebook installs all required libraries using:

```bash
!pip install --upgrade diffusers transformers accelerate torch bitsandbytes scipy safetensors xformers
```

You don’t need to do anything else — the notebook handles setup automatically.

## 📦 Model Details

- **Model ID:** `stabilityai/stable-diffusion-2-1`

### Libraries Used:

- `diffusers`
- `transformers`
- `torch` with CUDA
- `matplotlib` for image visualization


## 🧪 Sample Code

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "kid playing as iron man and defeating thanos"
image = pipe(prompt, width=1000, height=1000).images[0]

plt.imshow(image)
plt.axis('off')
plt.show()
```

## 🖼️ Output

The above example generates an image of a kid playing Iron Man defeating Thanos — but you can change the prompt to whatever you like!

## 🤝 Contributing

Feel free to fork and enhance this notebook! You can:

- Try different schedulers or samplers  
- Modify image resolution and prompt settings  
- Add prompt loops or batch generation  
- Export and save images automatically  

## 📄 License

This project is open-source under the MIT License.

