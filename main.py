from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

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
