"""
Imager - Image Processing API
Transforms source images to resemble target images by pixel rearrangement only.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
from typing import Optional

from processor import ImageProcessor
from image_utils import load_image, get_image_info

app = FastAPI(
    title="Imager API",
    description="Transform images through pixel rearrangement",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = ImageProcessor()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Imager</title>
            <style>
                body {
                    font-family: system-ui, -apple-system, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 { color: #333; }
                code {
                    background: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
                .endpoint {
                    background: #f9f9f9;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid #007bff;
                }
            </style>
        </head>
        <body>
            <h1>🖼️ Imager API</h1>
            <p>Transforms images by pixel rearrangement.</p>
            
            <h2>Endpoints</h2>
            
            <div class="endpoint">
                <strong>POST /process</strong><br>
                Upload source and target image.<br>
                <code>Form data: source_image, target_image</code>
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong><br>
                Check API status
            </div>
            
            <p>📚 <a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "imager-api",
        "version": "0.1.0"
    }


@app.post("/process")
async def process_images(
    source_image: UploadFile = File(..., description="Source image to rearrange"),
    target_image: UploadFile = File(..., description="Target image to match"),
    method: Optional[str] = "simple"
):
    try:
        source_bytes = await source_image.read()
        target_bytes = await target_image.read()
        
        source_img = load_image(source_bytes, mode='RGB')
        target_img = load_image(target_bytes, mode='RGB')
        
        output_img = processor.process(source_img, target_img, method=method)
        
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=output.png"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing images: {str(e)}"
        )


@app.post("/analyze")
async def analyze_images(
    source_image: UploadFile = File(..., description="Source image"),
    target_image: UploadFile = File(..., description="Target image")
):
    try:
        source_bytes = await source_image.read()
        target_bytes = await target_image.read()
        
        source_img = load_image(source_bytes, mode='RGB')
        target_img = load_image(target_bytes, mode='RGB')
        
        source_info = get_image_info(source_img)
        target_info = get_image_info(target_img)
        
        return {
            "source": {
                "width": source_info['width'],
                "height": source_info['height'],
                "pixels": source_info['pixel_count'],
                "mean_value": float(source_info['mean_value']),
                "std_value": float(source_info['std_value']),
                "min_value": int(source_info['min_value']),
                "max_value": int(source_info['max_value'])
            },
            "target": {
                "width": target_info['width'],
                "height": target_info['height'],
                "pixels": target_info['pixel_count'],
                "mean_value": float(target_info['mean_value']),
                "std_value": float(target_info['std_value']),
                "min_value": int(target_info['min_value']),
                "max_value": int(target_info['max_value'])
            },
            "compatibility": {
                "same_dimensions": (source_info['width'] == target_info['width'] and 
                                   source_info['height'] == target_info['height']),
                "same_pixel_count": (source_info['pixel_count'] == target_info['pixel_count']),
                "note": "Images will be resized to same dimensions during processing"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing images: {str(e)}"
        )


if __name__ == "__main__":
    print("Starting Imager...")
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
