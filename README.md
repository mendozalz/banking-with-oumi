# Guía Práctica de Fine-Tuning: Mi Primer SLM Open-Source para Soporte Bancario (y Cómo Superé los Límites de la Nube)

Como estudiante apasionado por la ciencia de datos y la seguridad informática, siempre busco poner a prueba mi conocimiento teórico con retos del mundo real. Leer sobre grandes modelos de lenguaje es fascinante, pero la verdadera ingeniería ocurre cuando te ensucias las manos limpiando datos, ajustando parámetros y lidiando con restricciones de infraestructura.

Recientemente, decidí seguir un tutorial para construir un Modelo de Lenguaje Pequeño (SLM) especializado en clasificar intenciones de soporte técnico bancario. El objetivo era demostrar que un modelo pequeño y especializado puede ser extremadamente potente y preciso.

Si bien el tutorial del autor original es una excelente fuente de inspiración, mi camino para replicarlo no fue una línea recta. Me topé con problemas de dependencias, límites de cuota gratuita y consejos técnicos ambiguos. 

**Este tutorial no es solo una réplica; es la versión honesta y práctica, con las soluciones que a mí me funcionaron para lograr una Prueba de Concepto (PoC) funcional y completamente gratuita.**

---

## 🚀 El Objetivo Real

**Lo fundamental de este ejercicio es que domines el ciclo de vida del dato:** limpiar la información cruda, estructurarla en archivos `.jsonl`, inyectar el contexto de la tarea (instrucciones), ejecutar el ajuste fino supervisado (SFT) y entender cómo evaluarlo objetivamente usando un enfoque de "LLM-as-a-Judge".

**Puedes saltarte esa comparativa comercial y centrarte en el verdadero valor técnico: crear tu propio modelo de IA open-source.** Para correr pruebas de referencia contra modelos comerciales como GPT-5.4, tendrías que conectar tus propias API Keys y generar cobros por cada token procesado. Mi enfoque es maximizar el aprendizaje minimizando los costos.

---

## 🛠️ Fase 1: Ingeniería de Datos Local (En VS Code)

El dataset de origen es el clásico *Banking77* de Hugging Face, que contiene más de 10,000 consultas de usuarios reales clasificadas en 77 categorías de intención.

### El Requisito de OUMI

Para entrenar en la plataforma OUMI, necesitamos convertir los datos crudos a un formato `.jsonl` (JSON Lines). Cada línea debe ser un objeto JSON completo con una estructura específica de "Instrucción -> Input -> Output". Además, debemos mapear los IDs numéricos de las categorías (0-76) a sus nombres de texto reales (ej. `card_arrival`).

```json
{
  "instruction": "Clasifica la siguiente consulta bancaria en una de las 77 categorías de intención.",
  "input": "Mi tarjeta no ha llegado",
  "output": "card_arrival"
}
```

### 🛠️ El Desafío Técnico y la Solución
Al ejecutar el script de preprocesamiento, me topé con dos obstáculos mayores:

1.  **Dependency Hell**: Las versiones modernas de la librería `datasets` de Hugging Face desactivaron la ejecución de scripts remotos por motivos de seguridad. Tuve que forzar una versión anterior (`pip install "datasets==2.19.2"`) para poder descargar los datos originales.
2.  **Límites de la Nube (Free Tier)**: El dataset tiene más de 10,000 filas. Al intentar entrenar en OUMI, la plataforma me bloqueó porque la cuota gratuita tiene un límite estricto de 1,000 filas en total (entrenamiento + validación).

### 🐍 El Script de Python que Funciona
Para superar esto, modifiqué el pipeline en Python para hacer un recorte inteligente (*subset*) de los datos. Mi script final toma aleatoriamente **800 ejemplos para entrenamiento** y **200 ejemplos para prueba**, manteniéndonos exactamente en el límite de las 1,000 filas.

He aquí el script completo que utilicé en VS Code (`preprocess_banking77.py`):

```
import json
import os
import random
from datasets import load_dataset

# Crear carpeta de proyecto
os.makedirs("banking-with-oumi", exist_ok=True)

# 1. Cargar dataset (forzar trust_remote_code si usasdatasets<=2.19.2)
# O usa dataset = load_dataset("mteb/banking77") como alternativa segura.
dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
label_names = dataset['train'].features['label'].names

def get_intent_name(label_id):
    return label_names[label_id]

# Instrucción contextual base
base_instruction = "Clasifica la siguiente consulta de soporte técnico bancario en la categoría de intención correcta. Tu salida debe ser únicamente el nombre de la categoría."

# 2. Generar todos los datos estructurados
all_train_data = [{"instruction": base_instruction, "input": row['text'].strip(), "output": get_intent_name(row['label'])} for row in dataset['train']]
all_test_data = [{"instruction": base_instruction, "input": row['text'].strip(), "output": get_intent_name(row['label'])} for row in dataset['test']]

# ---------------------------------------------------------
# 🛠️ SOLUCIÓN PARA LÍMITES DE CUOTA (FREE TIER OUMI)
# ---------------------------------------------------------
random.seed(42) # Semilla fija para reproducibilidad
random.shuffle(all_train_data)
random.shuffle(all_test_data)

# Tomamos 800 para train y 200 para test (Total = 1000 filas)
mini_train_data = all_train_data[:800]
mini_test_data = all_test_data[:200]

print(f"✅ Filas para entrenamiento: {len(mini_train_data)}")
print(f"✅ Filas para validación/test: {len(mini_test_data)}")
# ---------------------------------------------------------

# 3. Función para guardar
def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 4. Guardar los archivos reducidos
print("💾 Guardando archivos reducidos para Oumi Free Tier...")
save_jsonl(mini_train_data, "banking-with-oumi/Banking77-mini-train.jsonl")
save_jsonl(mini_test_data, "banking-with-oumi/banking77-mini-test.jsonl")

print("🎉 ¡Listo! Tienes los archivos 'mini' necesarios.")
```

## ☁️ Fase 2: Fine-Tuning Supervisado (SFT) en OUMI
Una vez que generé localmente mis archivos mini y los subí a la plataforma, pasé a la fase de entrenamiento.

> [!IMPORTANT]
> **Cuidado con el "Data Leakage"**
> Aquí es donde el tutorial del autor original genera confusión. Sugiere que puedes pedirle al agente de OUMI: "ajustar Qwen3.5 en el conjunto de prueba".
>
> **Riesgo Crítico**: Entrenar (*fine-tuning*) usando un conjunto de prueba (*test set*) es uno de los mayores errores en Machine Learning. Se conoce como filtración de datos (*data leakage*). Si dejas que tu modelo estudie con las respuestas del examen antes de tomarlo, creerás que es perfecto, pero cuando lo despliegues en el mundo real, fallará estrepitosamente. Siempre debes mantener tu dataset de validación/prueba completamente aislado.

### ⚙️ Configuración del Entrenamiento
Elegí el modelo **Qwen/Qwen3.5-0.8B** por ser ligero, rápido y perfecto para una PoC gratuita. Mi configuración final fue:

*   **Modelo Base**: `Qwen/Qwen3.5-0.8B`
*   **Método**: SFT (*Supervised Fine-Tuning*)
*   **Estrategia**: FFT (*Full Fine-Tuning* - ajustando todos los pesos neuronales)
*   **Dataset Entrenamiento**: `Banking77-mini-train.jsonl` (800 filas)
*   **Dataset Validación**: `banking77-mini-test.jsonl` (200 filas)

El entrenamiento finalizó con éxito, logrando una reducción dramática en la función de pérdida (*loss*), lo que indica que el modelo absorbió los patrones correctamente.

---

## 📊 Fase 3: Evaluación Objetiva (LLM-as-a-Judge)
No podemos evaluar la precisión de un clasificador de 77 categorías "a ojo". Implementé un método automático tipo **"LLM-as-a-Judge"**, configurando un prompt para que un modelo superior calificara las respuestas de mi Qwen contra las respuestas correctas del set de prueba.

### 🏆 Resultados de mi Prueba de Concepto (PoC)
Tras configurar los evaluadores y usar **DeepSeek-V3.1** como juez (mucho más preciso siguiendo instrucciones de estructura que Llama-3.3-70B), obtuve estos resultados:

*   ✅ **Validity (Validez)**: **~95-97%**. El modelo aprendió perfectamente a responder solo con la etiqueta, sin texto extra. Crucial para automatización.
*   📈 **Accuracy (Exactitud)**: **~11-27%**. Resultado excelente considerando que solo usamos 800 filas (~10 ejemplos por categoría). Demuestra aprendizaje de patrones real.

### 🎤 ¿Qué Sigue?
He logrado crear un prototipo técnico funcional superando las barreras iniciales. El objetivo final es descargar los pesos (*weights*) a mi entorno local de Ubuntu y conectarlo para construir un agente de voz automatizado. ¡Nos vemos en la Parte 2!

