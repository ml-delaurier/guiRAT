# 🧠 guiRAT 🐀 (Graphical Retrieval Augmented Thinking)

guiRAT is a GUI fork of the original RAT project. Maintaining the core concept of two-stage reasoning, it adds a user-friendly interface and integrates Groq.

![guiRAT Interface](assets/guiRAT-0.webp)

*because DeepSeek API is down...*
> *DeepSeek-R1 `deepseek-reasoner`
will be avaliable ASAP. 

> Currently using `groq-r1-distill-llama-70b` for the response stage.


> The original RAT concept was developed by skirano, as detailed in this [Twitter thread](https://x.com/skirano/status/1881922469411643413).



## How It Works

1. Magic

![Reasoning Stage](assets/guiRAT-1.webp)


## 🎯 Features

- 🌓 **Dark Mode**: No other choice


![Features Overview](assets/guiRAT-2.webp)

## ⚙️ Requirements

• Python 3.11 or higher  
• A .env file containing:

  ```plaintext
  GROQ_API_KEY=your_groq_api_key_here
  ```

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ml-delaurier/guiRAT.git
   cd guiRAT
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

1. Ensure your .env file is configured with the required API keys
2. Launch the GUI application:

   ```bash
   python rat_gui.py
   ```
---

![Usage Example](assets/guiRAT-3.webp)
---

## Default Configuration
- Default Reasoning Model: `deepseek-r1-distill-llama-70b`
- Default Base Model: `llama-3.3-70b-versatile`

All models listed above can be used in both reasoning and base roles, allowing for flexible configuration based on your specific needs.

### 🤖 Available Models

### DeepSeek Models
- **DeepSeek R1 Distill 70B**
  - Description: Specialized model for generating detailed reasoning
  - Context Length: 128,000 tokens

### Google Models
- **Gemma 2 9B IT**
  - Description: Google's instruction-tuned 9B parameter model
  - Context Length: 8,192 tokens

### Meta (Llama) Models
- **Llama 3.3 70B Versatile**
  - Description: Meta's versatile large language model with extended context
  - Context Length: 128,000 tokens
- **Llama 3.1 8B Instant**
  - Description: Fast and efficient 8B parameter model with extended context
  - Context Length: 128,000 tokens
- **Llama Guard 3 8B**
  - Description: Meta's safety-focused model with 8B parameters
  - Context Length: 8,192 tokens
- **Llama 3 70B**
  - Description: Meta's powerful 70B parameter model
  - Context Length: 8,192 tokens
- **Llama 3 8B**
  - Description: Efficient 8B parameter model for general use
  - Context Length: 8,192 tokens
- **Llama 3.3 70B SpecDec**
  - Description: Meta's specialized decoder variant of the 70B model
  - Context Length: 8,192 tokens

### Preview Models
- **Llama 3.2 1B Preview**
  - Description: Lightweight preview model with extended context
  - Context Length: 128,000 tokens
- **Llama 3.2 3B Preview**
  - Description: Medium-sized preview model with extended context
  - Context Length: 128,000 tokens
- **Llama 3.2 11B Vision Preview**
  - Description: Vision-capable preview model with extended context
  - Context Length: 128,000 tokens
- **Llama 3.2 90B Vision Preview**
  - Description: Large vision-capable preview model with extended context
  - Context Length: 128,000 tokens

### Mistral Models
- **Mixtral 8x7B**
  - Description: Mistral's mixture of experts model with extended context
  - Context Length: 32,768 tokens


