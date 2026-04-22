import json
import os
import random
from datasets import load_dataset

os.makedirs("banking-with-oumi", exist_ok=True)

# 1. Cargar el dataset (usando la versión compatible que encontramos)
dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
label_names = dataset['train'].features['label'].names

def get_intent_name(label_id):
    return label_names[label_id]

base_instruction = "Clasifica la siguiente consulta de soporte técnico bancario en la categoría de intención correcta. Tu salida debe ser únicamente el nombre de la categoría."

# 2. Generar todos los datos estructurados
all_train_data = [{"instruction": base_instruction, "input": row['text'].strip(), "output": get_intent_name(row['label'])} for row in dataset['train']]
all_test_data = [{"instruction": base_instruction, "input": row['text'].strip(), "output": get_intent_name(row['label'])} for row in dataset['test']]

# ---------------------------------------------------------
# NUEVO: RECORTE ESTRICTO PARA EVITAR LÍMITES DE CUOTA
# ---------------------------------------------------------
# Mezclamos aleatoriamente para no sesgar el modelo
random.seed(42) # Fijamos semilla para resultados reproducibles
random.shuffle(all_train_data)
random.shuffle(all_test_data)

# Tomamos 800 para train y 200 para test (Total = 1000 filas)
mini_train_data = all_train_data[:800]
mini_test_data = all_test_data[:200]

print(f"Filas para entrenamiento: {len(mini_train_data)}")
print(f"Filas para validación/test: {len(mini_test_data)}")
# ---------------------------------------------------------

# 3. Función para guardar
def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 4. Guardar los archivos reducidos
print("Guardando archivos reducidos para Oumi Free Tier...")
save_jsonl(mini_train_data, "banking-with-oumi/Banking77-mini-train.jsonl")
save_jsonl(mini_test_data, "banking-with-oumi/banking77-mini-test.jsonl")

print("¡Listo! Sube estos dos archivos y lánzalo.")