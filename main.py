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
        
        # Open as PIL Images
        source_img = Image.open(io.BytesIO(source_bytes)).convert('RGB')
        target_img = Image.open(io.BytesIO(target_bytes)).convert('RGB')
        
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
        
        # Open as PIL Images
        source_img = Image.open(io.BytesIO(source_bytes)).convert('RGB')
        target_img = Image.open(io.BytesIO(target_bytes)).convert('RGB')
        
        source_arr = np.array(source_img)
        target_arr = np.array(target_img)
        
        return {
            "source": {
                "width": source_img.width,
                "height": source_img.height,
                "pixels": source_img.width * source_img.height,
                "mean_color": source_arr.mean(axis=(0, 1)).tolist(),
                "std_color": source_arr.std(axis=(0, 1)).tolist()
            },
            "target": {
                "width": target_img.width,
                "height": target_img.height,
                "pixels": target_img.width * target_img.height,
                "mean_color": target_arr.mean(axis=(0, 1)).tolist(),
                "std_color": target_arr.std(axis=(0, 1)).tolist()
            },
            "compatibility": {
                "same_dimensions": (source_img.width == target_img.width and 
                                   source_img.height == target_img.height),
                "same_pixel_count": (source_img.width * source_img.height == 
                                    target_img.width * target_img.height)
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
