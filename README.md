# 🧠 guiRAT (Graphical Retrieval Augmented Thinking)

guiRAT is a GUI fork of the original RAT project. While maintaining the core concept of two-stage reasoning, it adds a user-friendly interface built with CustomTkinter and integrates Groq's powerful language models for enhanced performance

![guiRAT Interface](assets/guiRAT.webp)

*because DeepSeek API is down...*
> *DeepSeek-R1 `deepseek-reasoner`
will be avaliable ASAP. 

> Currently using `groq-r1-distill-llama-70b` for the response stage.

> Base models include:
   ```plaintext
   llama-3.3-70b-specdec
   llama-3.2-1b-preview
   llama-3.2-3b-preview
   llama-3.2-11b-vision
   llama-3.2-90b-vision-instruct
   ```

The original RAT concept was developed by skirano, as detailed in this [Twitter thread](https://x.com/skirano/status/1881922469411643413).



## How It Works

guiRAT employs a two-stage approach with a modern GUI interface:
1. **Reasoning Stage** (DeepSeek): Generates detailed reasoning and analysis for each query
2. **Response Stage** (Groq): Utilizes the reasoning context to provide informed, well-structured answers
3. **Interface**: Modern dark-themed GUI with conversation bubbles, model selection, and real-time responses

![Reasoning Stage](assets/guiRAT-001.webp)

This approach ensures more thoughtful, contextually aware, and reliable responses while providing an intuitive user experience.

## 🎯 Features

- 🖥️ **Modern GUI Interface**: Built with CustomTkinter for a sleek, user-friendly experience
- 🤖 **Groq Integration**: High-performance language model responses
- 🧠 **Reasoning Visibility**: Toggle visibility of the AI's thinking process
- 🌓 **Dark Mode**: Comfortable dark theme for extended use
- 💬 **Conversation History**: Maintains and displays full conversation context
- 🔄 **Real-time Updates**: Stream responses as they're generated

![Features Overview](assets/guiRAT-002.webp)

## ⚙️ Requirements

• Python 3.11 or higher  
• A .env file containing:
  ```plaintext
  DEEPSEEK_API_KEY=your_deepseek_api_key_here
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

## 📖 Usage

1. Ensure your .env file is configured with the required API keys
2. Launch the GUI application:
   ```bash
   python rat_gui.py
   ```

3. Features available in the GUI:
   - Type your questions in the input field
   - Select different models from the dropdown menus
   - Toggle reasoning visibility
   - Clear conversation history
   - Dark mode interface for comfortable use

---

![Usage Example](assets/guiRAT-003.webp)
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


