# main.py
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random
import os
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
import json
from pydantic import BaseModel
from typing import Optional
import requests
import os
from datetime import datetime
from gradio_client import Client
import shutil
from description_generator import DescriptionGenerator
import numpy as np
from scipy.stats import truncnorm


app = FastAPI(title="Yu-Gi-Oh Card Pack Generator")

# Create directories if they don't exist
os.makedirs("static/images", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

desc_model = DescriptionGenerator()

# Card model
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


async def generate_dragon_image(prompt: str, dragon_name: str, num_inference_steps: int = 50, guidance_scale: float = 7.5) -> str:
    """
    Generate a dragon image using your Hugging Face Space and save it locally
    """
    
    try:
        # Connect to your Hugging Face Space
        client = Client("marshad/yugioh-image")
        
        print(f"Generating image with prompt: {prompt}")
        
        # Make prediction using your space's API
        result = client.predict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            api_name="/generate_image"
        )
        
        print(f"API Result: {result}")
        
        # Create directory if it doesn't exist
        os.makedirs("static/images/dragons", exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in dragon_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_name:  # If dragon_name is empty or invalid
            safe_name = "dragon"
        
        # Keep the original extension (.webp) or convert to .png
        filename = f"{safe_name}_{timestamp}.webp"  # Keep as webp
        filepath = f"static/images/dragons/{filename}"
        
        # The result is directly a file path string
        source_path = result
        
        print(f"Source path: {source_path}")
        
        if source_path and os.path.exists(source_path):
            # Copy the generated image to your static directory
            shutil.copy(source_path, filepath)
            print(f"Image saved to: {filepath}")
            return filepath
        else:
            print(f"Source path does not exist: {source_path}")
            raise Exception
            
    except Exception as e:
        print(f"Error generating dragon image: {str(e)}")
        print(f"Full error details: {repr(e)}")
        raise Exception
    


async def generate_dragon_image_as_png(prompt: str, dragon_name: str, num_inference_steps: int = 50, guidance_scale: float = 7.5) -> str:
    """
    Same as above but converts webp to png
    """
    try:
        from PIL import Image
        
        # First generate the webp image
        webp_path = await generate_dragon_image(prompt, dragon_name, num_inference_steps, guidance_scale)
        
        if webp_path.endswith('.webp'):
            # Convert to PNG
            png_path = webp_path.replace('.webp', '.png')
            
            # Open and convert
            with Image.open(webp_path) as img:
                img.save(png_path, 'PNG')
            
            # Remove the webp file
            os.remove(webp_path)
            
            print(f"Converted to PNG: {png_path}")
            return png_path
        
        return webp_path
        
    except Exception as e:
        print(f"Error converting to PNG: {e}")
        return webp_path  # Return original if conversion fails
    
def atk_def_sample():
    # Parameters
    lower, upper = 0, 4000
    mean = 2500
    std = 400  # Estimated standard deviation

    # Convert bounds to standard normal space (a, b)
    a, b = (lower - mean) / std, (upper - mean) / std

    # Create a truncated normal distribution
    dist = truncnorm(a, b, loc=mean, scale=std)

    # Sample once and round to nearest 10
    atk = round(dist.rvs() / 10) * 10
    def_ = round(dist.rvs() / 10) * 10

    return int(atk), int(def_)


# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/generate-dragon")
async def generate_dragon(
    dragon_name: str = Query(..., description="Name of the dragon"),
    image_prompt: str = Query(..., description="Prompt for image generation"),
    num_inference_steps = 25,
    guidance_scale = 7.5
):
    try:
        print(f"Generating dragon: {dragon_name}")
        print(f"With prompt: {image_prompt}")
        
        # Generate the dragon image
        image_path = await generate_dragon_image(
            prompt=image_prompt, 
            dragon_name=dragon_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        description = desc_model.generate_description(dragon_name)
        atk, def_ = atk_def_sample()
        dragon_card = {
            "name": dragon_name,
            "type": "monster",
            "description": description,
            "attack": atk,
            "defense": def_,
            "image_url": f"/{image_path}"  # Serve as static file
        }
        
        return JSONResponse(content={
            "success": True,
            "dragon": dragon_card,
            "image_path": image_path
        })
    
    except Exception as e:
        print(f"Error in generate_dragon route: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)