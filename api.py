from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import base64
from waste_bin_monitor import calculate_fill_level

app = FastAPI()

# Allow requests from our React frontend (http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-bin")
async def analyze_bin(image: UploadFile = File(...)):
    temp_input_path = f"temp_{image.filename}"
    temp_output_path = f"result_{image.filename}.jpg"

    try:
        # Save uploaded image
        with open(temp_input_path, "wb+") as f:
            f.write(await image.read())

        # Process the image using our algorithm
        # We save the output image but don't show the OpenCV window
        fill_percentage = calculate_fill_level(
            image_path=temp_input_path,
            save_path=temp_output_path,
            show_window=False
        )

        # Read the generated result image and convert to Base64 to send to React
        with open(temp_output_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
        return {
            "success": True,
            "fillLevel": round(fill_percentage, 1),
            "resultImage": f"data:image/jpeg;base64,{encoded_string}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

if __name__ == "__main__":
    print("Starting Waste Bin Monitor API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
