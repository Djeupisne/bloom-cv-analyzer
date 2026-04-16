from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber

app = FastAPI()

# Charger Bloom (version 7B pour commencer)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")

class CVRequest(BaseModel):
    cv_text: str

@app.post("/analyze")
def analyze_cv(request: CVRequest):
    inputs = tokenizer(request.cv_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300)
    result = tokenizer.decode(outputs[0])
    return {"analysis": result}

@app.post("/upload")
async def upload_cv(file: UploadFile = File(...)):
    # Extraire le texte d'un PDF
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    else:
        return {"error": "Seuls les fichiers PDF sont supportés pour l'instant."}

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300)
    result = tokenizer.decode(outputs[0])
    return {"analysis": result}
