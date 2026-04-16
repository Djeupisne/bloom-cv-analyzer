from transformers import AutoTokenizer, AutoModelForCausalLM

# Charger Bloom (version plus légère pour commencer)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")

# Exemple d'analyse simple
text = "Analyse ce CV : Ingénieur logiciel avec 5 ans d'expérience en banque..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)

print(tokenizer.decode(outputs[0]))
