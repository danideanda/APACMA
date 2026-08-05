import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from huggingface_hub import login
import os

def verificar_ruta_modelo():
    global model_path
    # ========== verificar ruta del modelo ==========
    if os.path.exists("models/LLM"):
        model_path = "models/LLM"
    elif os.path.exists("models/LLM-base"):
        model_path = "models/LLM-base"
    else:
        model_path = "error fatal: no se encontró la ruta del modelo"

def entrenar_fine(modelo_path, dataset_path="./dataset.json", output_dir="./fine_tuned_model"):
    """
    Función para realizar fine-tuning con LoRA de un modelo base
    
    Args:
        modelo_path (str): Ruta al modelo base o nombre en HuggingFace
        dataset_path (str): Ruta al archivo dataset.json (por defecto "./dataset.json")
        output_dir (str): Directorio donde guardar el modelo fine-tuneado
    
    Returns:
        str: Ruta del modelo guardado
    """
    
    # Verificar que existe el archivo de datos
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"El archivo {dataset_path} no existe")
    
    # Cargar el dataset
    print(f"Cargando datos desde {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Si es un solo objeto, convertirlo a lista
    if isinstance(data, dict):
        data = [data]
    
    # Preparar los datos para el formato de entrenamiento
    training_data = []
    for item in data:
        # Crear el prompt con el formato adecuado para el modelo
        prompt = f"Pregunta: {item['input']}\nRespuesta: {item['output']}"
        training_data.append({"text": prompt})
    
    # Crear dataset de HuggingFace
    dataset = Dataset.from_list(training_data)
    
    # Dividir en train y eval (90% train, 10% eval)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Dataset cargado: {len(train_dataset)} ejemplos de entrenamiento, {len(eval_dataset)} de evaluación")
    
    # Cargar tokenizador y modelo
    print(f"Cargando modelo desde {modelo_path}...")
    tokenizer = AutoTokenizer.from_pretrained(modelo_path)
    model = AutoModelForCausalLM.from_pretrained(
        modelo_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configurar tokenizador
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Preparar modelo para entrenamiento con k-bit
    model = prepare_model_for_kbit_training(model)
    
    # Configuración de LoRA
    lora_config = LoraConfig(
        r=8,  # Dimensión de la matriz de adaptación
        lora_alpha=32,  # Factor de escala
        target_modules=["q_proj", "v_proj"],  # Módulos a adaptar
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Aplicar LoRA al modelo
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Función de tokenización
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # Tokenizar datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        learning_rate=2e-4,
        weight_decay=0.01,
        report_to=None,  # Desactivar wandb/tensorboard para este ejemplo
    )
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Entrenar
    print("Iniciando entrenamiento...")
    trainer.train()
    
    # Guardar el modelo fine-tuneado
    print(f"Guardando modelo en {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("¡Fine-tuning completado!")
    return output_dir