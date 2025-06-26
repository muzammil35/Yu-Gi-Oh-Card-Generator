# Yu-Gi-Oh Dragon Card Generator

A full-stack AI-powered web application that generates complete Yu-Gi-Oh dragon monster cards with original artwork, descriptions, and stats. This project combines fine-tuned machine learning models with a FastAPI backend and web interface to create an interactive card generation experience.

## 🎯 Project Overview

This comprehensive system generates authentic Yu-Gi-Oh dragon cards through:
- **Image Generation**: Fine-tuned Stable Diffusion XL model deployed on Hugging Face Spaces
- **Text Generation**: Fine-tuned GPT-2 model for creating card descriptions and lore
- **Web Application**: FastAPI backend with HTML frontend for interactive card generation
- **Statistical Modeling**: Truncated normal distribution for realistic ATK/DEF stat generation

## 🛠 Technical Architecture

### Backend Stack
- **FastAPI**: High-performance Python web framework
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and serialization
- **Jinja2**: Template engine for dynamic HTML rendering

### AI/ML Pipeline
- **Base Model**: Stable Diffusion XL (SDXL) from Stability AI
- **Fine-tuning Method**: DreamBooth with LoRA (Low-Rank Adaptation)
- **Deployment**: Hugging Face Spaces with Gradio Client integration
- **Text Generation**: Custom GPT-2 model via `DescriptionGenerator` class

### Frontend & Infrastructure
- **Static File Serving**: Integrated image and asset management
- **Real-time Generation**: Asynchronous image generation with progress tracking
- **Image Processing**: PIL-based WebP to PNG conversion pipeline
- **RESTful API**: Clean API endpoints for card generation

## 🚀 Key Features

### Core Functionality
- **Interactive Web Interface**: User-friendly card generation portal
- **Real-time AI Generation**: Live dragon artwork creation with customizable parameters
- **Complete Card Data**: Name, description, ATK/DEF stats, and high-quality artwork
- **Statistical Realism**: Scientifically modeled stat distribution (μ=2500, σ=400)

### Technical Capabilities
- **Hugging Face Integration**: Seamless connection to deployed SDXL model
- **Asynchronous Processing**: Non-blocking image generation for better UX
- **Error Handling**: Robust exception management and fallback mechanisms
- **File Management**: Automated image storage and serving system

## 📊 Model Specifications

### Stable Diffusion XL Fine-tuning Parameters
```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="/content/drive/MyDrive/dragons/images" \
  --output_dir="/content/dragons_lora" \
  --mixed_precision="fp16" \
  --instance_prompt="a detailed YU-GI-Oh dragon monster with intricate artwork" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --max_train_steps=400 \
  --checkpointing_steps=50 \
  --seed=0
```

### Statistical Modeling
```python
# ATK/DEF Generation using Truncated Normal Distribution
lower, upper = 0, 4000
mean = 2500
std = 400
# Rounded to nearest 10 for authentic Yu-Gi-Oh stat formatting
```

## 🏗 System Architecture

### API Endpoints
- **`GET /`**: Main web interface
- **`GET /api/generate-dragon`**: Dragon card generation endpoint
  - Parameters: `dragon_name`, `image_prompt`, `num_inference_steps`, `guidance_scale`
  - Returns: Complete card data with image URL

### Image Generation Pipeline
1. **Prompt Processing**: Custom prompt engineering for Yu-Gi-Oh style
2. **HF Spaces Integration**: Remote model inference via Gradio Client
3. **File Management**: Automated image download and storage
4. **Format Conversion**: WebP to PNG conversion with PIL
5. **Static Serving**: Direct image serving through FastAPI

### Data Models
```python
class Card(BaseModel):
    id: str
    name: str
    subtype: Optional[str] = None
    attribute: Optional[str] = None
    race: Optional[str] = None
    stars: Optional[int] = None
    attack: Optional[int] = None
    defense: Optional[int] = None
    description: str
    image_path: str
```

## 🎨 Generation Process

### Image Generation
- **Hugging Face Space**: `marshad/yugioh-image`
- **Configurable Parameters**: Inference steps (25-50), guidance scale (7.5)
- **Output Format**: High-resolution WebP/PNG images
- **Naming Convention**: `{dragon_name}_{timestamp}.{ext}`

### Text Generation
- **Custom Model**: Fine-tuned GPT-2 via `DescriptionGenerator`
- **Contextual Output**: Authentic Yu-Gi-Oh card descriptions
- **Integration**: Seamless combination with image generation

## 🔧 Technical Achievements

- **Full-Stack ML Application**: Complete web application with AI backend
- **Production Deployment**: FastAPI with proper async handling and error management
- **Model Integration**: Seamless connection between multiple AI services
- **Real-time Processing**: Efficient image generation and serving pipeline
- **Robust Architecture**: Comprehensive error handling and file management

## 📚 Dependencies

### Core Framework
- FastAPI, Uvicorn, Pydantic
- Jinja2 Templates, Static Files

### AI/ML Libraries
- Gradio Client (Hugging Face integration)
- Custom DescriptionGenerator module
- NumPy, SciPy (statistical modeling)

### Image Processing
- PIL (Python Imaging Library)
- WebP/PNG conversion pipeline

## 🚀 Usage

### Local Development
```bash
python main.py
# Server runs on http://0.0.0.0:8000
```

### API Usage
```bash
curl "http://localhost:8000/api/generate-dragon?dragon_name=Ancient%20Fire%20Dragon&image_prompt=a%20detailed%20YU-GI-Oh%20dragon%20monster%20with%20intricate%20artwork"
```

### Web Interface
Navigate to the root URL for the interactive card generation interface.

## 📈 Performance Features

- **Asynchronous Processing**: Non-blocking image generation
- **Efficient File Handling**: Optimized image storage and serving
- **Error Recovery**: Graceful handling of generation failures
- **Resource Management**: Automatic cleanup and file organization

## 📝 Future Enhancements

- **Database Integration**: Persistent card storage and retrieval
- **User Authentication**: Personal card collections
- **Batch Generation**: Multiple card generation
- **Export Features**: PDF/printable card formats
- **Advanced Customization**: More granular generation parameters
- **Card Rarity System**: Automatic rarity assignment and visual effects
