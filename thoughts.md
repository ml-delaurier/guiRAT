# RAT (Retrieval Augmented Thinking) Codebase Analysis

## Project Overview
RAT is a Python-based tool that enhances AI responses through a two-stage process:
1. Reasoning Stage (using DeepSeek)
2. Response Stage (using OpenRouter)

## Core Components

### ModelChain Class
- **Location**: `rat/rat.py`
- **Purpose**: Main orchestrator that manages the interaction between DeepSeek and OpenRouter APIs
- **Key Features**:
  - Maintains separate message histories for both models
  - Supports model switching via OpenRouter
  - Includes timing metrics for reasoning process
  - Toggleable reasoning visibility

### Key Dependencies
- OpenAI SDK for API interactions
- Rich library for terminal formatting
- Prompt_toolkit for interactive CLI
- python-dotenv for environment management
- pyperclip for clipboard operations

## Architecture Analysis

### Strengths
1. **Modular Design**: Clear separation between reasoning and response stages
2. **Flexible Model Selection**: Easy to switch between different OpenRouter models
3. **Interactive CLI**: Well-formatted terminal interface with rich text support
4. **Context Management**: Maintains conversation history for both models

### Areas for Potential Enhancement
1. **Error Handling**: Could benefit from more robust error handling for API failures
2. **Configuration Management**: Consider moving model constants to config file
3. **Message History Management**: Could add message history size limits
4. **Testing**: No visible test suite implementation

## Security Considerations
- Uses environment variables for API keys
- No hardcoded credentials
- API keys required: DeepSeek, OpenRouter, (optionally) Anthropic

## Development Setup
- Python 3.11+ required
- Uses uv for package management
- Local package installation supported

## Implementation Details

### Reasoning Process
```python
def get_deepseek_reasoning():
    # Uses DeepSeek's unique reasoning_content feature
    # Streams responses for real-time feedback
    # Includes timing metrics
```

### Response Generation
```python
def get_openrouter_response():
    # Combines user input with reasoning
    # Supports streaming responses
    # Maintains conversation context
```

## Next Steps & Recommendations
1. Consider adding unit tests
2. Implement message history cleanup
3. Add more robust error handling
4. Consider adding logging system
5. Document API response formats