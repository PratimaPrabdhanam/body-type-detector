from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from detector import detect_from_bytes
from suggestions import get_suggestions

app = FastAPI()

# Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    body_type = detect_from_bytes(image_bytes)
    suggestions = get_suggestions(body_type)
    return {
        "body_type": body_type,
        "suggestions": suggestions
    }
