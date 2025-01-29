# Groq API Implementation Guide

## Overview
This document outlines how to replace OpenRouter with Groq API in the RAT project. Groq offers high-performance language models with an API interface similar to OpenAI's.

## Setup

1. Install the Groq Python package:
```bash
pip install groq
```

2. Add Groq API key to `.env`:
```plaintext
GROQ_API_KEY=your_groq_api_key_here
```

## Implementation Changes

### 1. Update Imports
Replace OpenRouter imports with Groq:
```python
from groq import Groq
```

### 2. Client Initialization
Replace OpenRouter client with Groq client:
```python
self.groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
```

### 3. Model Configuration
Default model configuration:
```python
GROQ_MODEL = "llama-3.3-70b-versatile"  # Replace OPENROUTER_MODEL
```

### 4. Response Generation Function
Replace `get_openrouter_response` with:
```python
def get_groq_response(self, user_input, reasoning):
    combined_prompt = (
        f"<question>{user_input}</question>\n\n"
        f"<thinking>{reasoning}</thinking>\n\n"
    )
    
    self.groq_messages.append({"role": "user", "content": combined_prompt})
    
    try:
        completion = self.groq_client.chat.completions.create(
            model=self.current_model,
            messages=self.groq_messages,
            temperature=0.5,
            max_completion_tokens=1024,
            stream=True
        )
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                
        self.groq_messages.append({"role": "assistant", "content": full_response})
        return full_response
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None
```

### 5. Model Switching
Update model switching to use Groq models:
```python
AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Llama 70B Versatile",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    # Add other available Groq models as needed
}
```

## Key Differences from OpenRouter

1. **API Structure**: Groq uses a similar API structure to OpenAI, making migration straightforward
2. **Model Names**: Different model identifiers (e.g., "llama-3.3-70b-versatile" instead of OpenRouter's model paths)
3. **Token Limits**: Supports up to 32,768 tokens shared between prompt and completion
4. **Streaming**: Similar streaming implementation but with slightly different response structure

## Error Handling

Add specific error handling for Groq API:
```python
from groq.error import GroqError

try:
    # API call
except GroqError as e:
    print(f"Groq API Error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

## Implementation Steps

1. Update dependencies in `requirements.txt` or `pyproject.toml`
2. Add Groq API key to environment variables
3. Replace OpenRouter client initialization
4. Update model constants and available models
5. Replace response generation function
6. Update error handling
7. Test with different models and parameters

## Testing

Test the implementation with:
```python
client = Groq()
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Test message"}],
    stream=True
)