# Fine-Tuning Llama 2 with Hugging Face & QLoRA

A Google Colab project demonstrating how to fine-tune a Llama 2 causal language model on a medical terminology dataset using **LoRA/QLoRA**, Hugging Face Transformers, PEFT, TRL, and bitsandbytes.

The notebook was configured to run on a **T4 GPU with ~15 GB VRAM** and uses 4-bit quantization to reduce GPU memory usage.

## 🚀 Project Overview

The goal of this project is to adapt a pretrained Llama 2 model to a medical terminology dataset so that it can generate responses related to medical concepts.

### Model

- Base model: `aboonaji/llama2finetune-v2`
- Architecture: Llama 2 / Causal Language Model
- Quantization: 4-bit NF4
- Fine-tuning method: LoRA / QLoRA
- Compute device: GPU (Google Colab T4)

### Dataset

- Dataset: `aboonaji/wiki_medical_terms_llam2_format`
- Split used: `train`
- Text field: `text`
- Dataset contains formatted medical terminology examples suitable for causal language-model fine-tuning.

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face TRL
- PEFT
- bitsandbytes
- Accelerate
- Weights & Biases
- Google Colab

## 📦 Library Versions

The notebook pins the following versions:

```text
Transformers 4.46.3
PEFT         0.13.2
TRL          0.12.0
Accelerate   1.0.1
```

`bitsandbytes` is also used for 4-bit model quantization.

## 🔄 Workflow

The project follows these steps:

```text
Install Dependencies
        ↓
Load 4-bit Llama 2 Model
        ↓
Load Tokenizer
        ↓
Configure Training Arguments
        ↓
Configure LoRA
        ↓
Create SFTTrainer
        ↓
Fine-Tune Model
        ↓
Generate Medical Responses
```

## ⚙️ Model Loading

The model is loaded using 4-bit quantization:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

llama_model = AutoModelForCausalLM.from_pretrained(
    "aboonaji/llama2finetune-v2",
    quantization_config=bnb_config,
    device_map="auto"
)

llama_model.config.use_cache = False
```

4-bit quantization significantly reduces GPU memory requirements and makes Llama fine-tuning more practical on a T4 GPU.

## 🎯 LoRA Configuration

The project uses PEFT LoRA instead of updating all model parameters:

```python
LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
```

This allows the fine-tuning process to train a much smaller set of adapter parameters instead of the complete base model.

## 🏋️ Training Configuration

The final training configuration used:

```python
TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    fp16=True,
    gradient_checkpointing=True
)
```

### Why these settings?

- **Batch size = 1:** reduces GPU memory consumption.
- **Gradient accumulation = 4:** provides an effective batch size of 4 without loading four examples into VRAM simultaneously.
- **FP16:** reduces memory usage and improves GPU efficiency.
- **Gradient checkpointing:** trades some computation for lower memory usage.
- **100 steps:** keeps the demonstration/training run manageable on Colab.

## 📊 Training Result

The training run successfully completed all 100 steps:

```text
Global steps:        100
Training loss:       1.6631109619140625
Training runtime:    1066.0851 seconds
Training samples/sec: 0.375
Training steps/sec:   0.094
```

The run completed on a Google Colab T4 GPU after reducing the training memory requirements.

## 💬 Inference

After fine-tuning, the model can be used with a Hugging Face text-generation pipeline:

```python
user_prompt = "Please tell me about Ascariasis"

text_generation_pipeline = pipeline(
    task="text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    max_length=300
)

model_answer = text_generation_pipeline(
    f"[INST]{user_prompt}[/INST]"
)[0]["generated_text"]

print(model_answer)
```

### Example

**Prompt:**

```text
Please tell me about Ascariasis
```

The fine-tuned model generated a response describing ascariasis as a parasitic infection caused by *Ascaris lumbricoides*, followed by information about transmission, symptoms, and related medical details.

## 📈 Experiment Tracking

The training run was optionally tracked using **Weights & Biases (W&B)**.

W&B records included:

- Training loss
- Runtime
- Training steps
- Samples per second
- Run information

W&B is not required for the actual fine-tuning process and can be disabled if experiment tracking is not needed.

## ⚠️ Important Notes

### GPU Memory

A full-precision Llama model can exceed the VRAM available on a T4 GPU. This project therefore uses:

- 4-bit quantization
- LoRA
- Batch size 1
- Gradient accumulation
- Gradient checkpointing

These choices are important for fitting the training workload into approximately 15 GB of GPU VRAM.

### TRL API Warnings

The notebook produced deprecation warnings related to `dataset_text_field` and the older `SFTTrainer` argument style. These warnings did not prevent the training run from completing.

Future versions of TRL may require moving these settings into `SFTConfig`.

### Medical Information

This project is intended as a **technical fine-tuning demonstration**. Generated medical responses should not be treated as professional medical advice or used for diagnosis or treatment decisions.

## 📁 Suggested Repository Structure

```text
fine-tuning-llama-medical/
│
├── Fine-Tuning-LLMs.ipynb
├── README.md
├── requirements.txt
└── results/
```

## 🔮 Future Improvements

- Train for more steps/epochs with proper validation.
- Add a held-out evaluation dataset.
- Compare the base model against the fine-tuned model.
- Track validation loss and other evaluation metrics.
- Experiment with LoRA rank and alpha values.
- Move to the newer TRL `SFTConfig` API.
- Save and publish the trained LoRA adapter.
- Build a small web interface for interacting with the model.
- Evaluate generated medical responses for factuality and safety.

## 👨‍💻 Author

**Kartik Palan**

This project was created as a practical demonstration of LLM fine-tuning using Hugging Face, PEFT/LoRA, and Google Colab.
