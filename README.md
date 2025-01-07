pip install transformers torch

pip show transformers torch


python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B'); \
AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')"

model_name = "EleutherAI/gpt-neo-1.3B"

transformers-cli login

model_name = "EleutherAI/gpt-neo-1.3B"  # Weniger Speicherintensiv

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Debugging: Modellname prüfen
model_name = "EleutherAI/gpt-neo-2.7B"  # Alternativen: gpt-neo-1.3B
print(f"Lade Modell: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Modell und Tokenizer erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells oder Tokenizers: {e}")
    exit()

# Eingabe verarbeiten
def generate_code(prompt, max_length=200):
    try:
        print(f"Verarbeite Eingabe: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(inputs['input_ids'], max_length=max_length, temperature=0.7)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Fehler bei der Generierung: {e}")
        return None

# Beispiel-Eingabe
prompt = "Schreibe ein Python-Programm, das zwei Zahlen addiert:"
print(f"Prompt: {prompt}")

code = generate_code(prompt)
if code:
    print("Generierter Code:")
    print(code)
else:
    print("Fehler bei der Codegenerierung.")

import torch

print(torch.cuda.is_available())  # Gibt True aus, wenn GPU verfügbar ist

def addiere_zahlen(a, b):
    return a + b

print(addiere_zahlen(3, 5))
