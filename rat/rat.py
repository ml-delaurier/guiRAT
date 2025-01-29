"""
RAT (Retrieval Augmented Thinking) with Groq Integration
This module implements a powerful reasoning chain that combines DeepSeek's reasoning capabilities
with Groq's high-performance language models for enhanced AI responses.
"""

from openai import OpenAI
from groq import Groq
import os
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.syntax import Syntax
import pyperclip
import time

# Model Constants
DEEPSEEK_MODEL = "deepseek-reasoner"
GROQ_MODEL = "deepseek-r1-distill-llama-70b"  # Default model optimized for reasoning tasks

# Reasoning Format Options for controlling model output structure
REASONING_FORMATS = {
    "parsed": "Separates reasoning into a dedicated field",
    "raw": "Includes reasoning within think tags in content",
    "hidden": "Returns only the final answer"
}

# Comprehensive list of available Groq models with descriptions
AVAILABLE_MODELS = {
    # Production Models - Reasoning Optimized
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill 70B (Reasoning)",
    
    # Production Models - General Purpose
    "llama-3.3-70b-versatile": "Llama 3 70B Versatile",
    "llama-3.3-70b-instruct": "Llama 3 70B Instruct",
    "llama-3.3-8b-instruct": "Llama 3 8B Instruct",
    "mixtral-8x7b-32768": "Mixtral 8x7B 32K",
    "mixtral-8x7b-instruct": "Mixtral 8x7B Instruct",
    "gemma-2b-instruct": "Gemma 2B Instruct",
    "gemma-7b-instruct": "Gemma 7B Instruct",
    
    # Preview Models - Experimental features, not for production use
    "llama-3.2-1b": "Llama 3.2 1B (Preview)",
    "llama-3.2-3b": "Llama 3.2 3B (Preview)",
    "llama-3.2-11b-vision": "Llama 3.2 11B Vision (Preview)",
    "llama-3.2-90b-vision-instruct": "Llama 3.2 90B Vision Instruct (Preview)"
}

# Load environment variables for API keys
load_dotenv()

class ModelChain:
    """
    ModelChain orchestrates the interaction between DeepSeek's reasoning capabilities
    and Groq's high-performance language models. It manages message history,
    handles model switching, and controls reasoning visibility and format.
    """

    def __init__(self):
        """
        Initialize the ModelChain with both DeepSeek and Groq clients.
        Sets up message history tracking and default configurations.
        """
        # Initialize DeepSeek client for reasoning
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        # Initialize Groq client for response generation
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Message history tracking for both models
        self.deepseek_messages = []
        self.groq_messages = []
        
        # Configuration settings
        self.current_model = GROQ_MODEL
        self.show_reasoning = True  # Controls visibility of reasoning process
        self.reasoning_format = "raw"  # Controls how reasoning is presented in output

    def set_model(self, model_name):
        """
        Switch to a different Groq model if it exists in available models.
        
        Args:
            model_name (str): Name of the model to switch to
            
        Returns:
            bool: True if model was switched successfully, False if invalid model
        """
        if model_name in AVAILABLE_MODELS:
            self.current_model = model_name
            return True
        return False

    def get_model_display_name(self):
        """
        Get the human-readable name of the current model.
        
        Returns:
            str: Display name of the current model
        """
        return AVAILABLE_MODELS.get(self.current_model, self.current_model)

    def set_reasoning_format(self, format_name):
        """
        Set the reasoning output format if valid.
        
        Args:
            format_name (str): Format to set ('parsed', 'raw', or 'hidden')
            
        Returns:
            bool: True if format was set successfully, False if invalid format
        """
        if format_name in REASONING_FORMATS:
            self.reasoning_format = format_name
            return True
        return False

    def get_deepseek_reasoning(self, user_input):
        """
        Get step-by-step reasoning from Groq model for the given input.
        Tracks timing and displays reasoning process if enabled.
        
        Args:
            user_input (str): User's question or prompt
            
        Returns:
            str: Generated reasoning content
        """
        start_time = time.time()
        self.deepseek_messages.append({"role": "user", "content": user_input})
        
        if self.show_reasoning:
            rprint("\n[blue]Reasoning Process[/]")
        
        response = self.groq_client.chat.completions.create(
            model=self.current_model,
            messages=self.deepseek_messages,
            stream=True
        )

        reasoning_content = ""
        final_content = ""

        # Process streaming response
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                reasoning_content += content
                final_content += content
                if self.show_reasoning:
                    print(content, end="", flush=True)

        # Display timing information
        elapsed_time = time.time() - start_time
        time_str = f"{elapsed_time/60:.1f} minutes" if elapsed_time >= 60 else f"{elapsed_time:.1f} seconds"
        rprint(f"\n\n[yellow]Thought for {time_str}[/]")
        
        if self.show_reasoning:
            print("\n")
        return reasoning_content

    def get_groq_response(self, user_input, reasoning):
        """
        Generate a response using Groq's API with optimized reasoning parameters.
        Combines user input with reasoning for context-aware responses.
        
        Args:
            user_input (str): Original user question
            reasoning (str): Generated reasoning from DeepSeek
            
        Returns:
            str: Generated response or None if error occurs
        """
        # Combine input and reasoning in structured format
        combined_prompt = (
            f"<question>{user_input}</question>\n\n"
            f"<thinking>{reasoning}</thinking>\n\n"
        )
        
        self.groq_messages.append({"role": "user", "content": combined_prompt})
        rprint(f"[green]{self.get_model_display_name()}[/]")
        
        try:
            # Create completion with optimized parameters for reasoning
            completion = self.groq_client.chat.completions.create(
                model=self.current_model,
                messages=self.groq_messages,
                temperature=0.6,  # Balanced setting for reasoning tasks
                max_completion_tokens=4096,  # Increased for complex reasoning chains
                top_p=0.95,  # Recommended for maintaining output quality
                stream=True,  # Enable streaming for real-time response
                stop=None,  # No custom stop sequences
                presence_penalty=0.1,  # Slight penalty to reduce repetition
                frequency_penalty=0.1,  # Encourage vocabulary diversity
                reasoning_format=self.reasoning_format,  # Control reasoning presentation
                seed=None  # Optional: Set for reproducible results
            )
            
            # Process streaming response
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
                    
        except Exception as e:
            rprint(f"\n[red]Error: {str(e)}[/]")
            return None
        
        # Update message history for both models
        self.deepseek_messages.append({"role": "assistant", "content": full_response})
        self.groq_messages.append({"role": "assistant", "content": full_response})
        
        print("\n")
        return full_response

def main():
    chain = ModelChain()
    
    # Initialize prompt session with styling
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    rprint(Panel.fit(
        "[bold cyan]Retrival augmented thinking[/]",
        title="[bold cyan]RAT [/]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'model <n>'[/] to change the Groq model")
    rprint(" • Type [bold magenta]'models'[/] to list available models")
    rprint(" • Type [bold magenta]'reasoning'[/] to toggle reasoning visibility")
    rprint(" • Type [bold magenta]'format <type>'[/] to change reasoning format (parsed/raw/hidden)")
    rprint(" • Type [bold magenta]'clear'[/] to clear chat history\n")
    
    while True:
        try:
            user_input = session.prompt("\nYou: ", style=style).strip()
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! ")
                break

            if user_input.lower() == 'clear':
                chain.deepseek_messages = []
                chain.groq_messages = []
                rprint("\n[magenta]Chat history cleared![/]\n")
                continue
                
            if user_input.lower().startswith('model '):
                new_model = user_input[6:].strip()
                if chain.set_model(new_model):
                    print(f"\nChanged model to: {chain.get_model_display_name()}\n")
                else:
                    rprint("\n[red]Invalid model. Use 'models' command to see available models.[/]\n")
                continue

            if user_input.lower() == 'models':
                rprint("\n[yellow]Available Models:[/]")
                for model_id, display_name in AVAILABLE_MODELS.items():
                    rprint(f" • [bold magenta]{model_id}[/]: {display_name}")
                print()
                continue

            if user_input.lower().startswith('format '):
                new_format = user_input[7:].strip()
                if chain.set_reasoning_format(new_format):
                    rprint(f"\n[magenta]Reasoning format set to: {new_format} - {REASONING_FORMATS[new_format]}[/]\n")
                else:
                    rprint("\n[red]Invalid format. Available formats: parsed, raw, hidden[/]\n")
                continue

            if user_input.lower() == 'reasoning':
                chain.show_reasoning = not chain.show_reasoning
                status = "visible" if chain.show_reasoning else "hidden"
                rprint(f"\n[magenta]Reasoning process is now {status}[/]\n")
                continue
            
            reasoning = chain.get_deepseek_reasoning(user_input)
            response = chain.get_groq_response(user_input, reasoning)
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()