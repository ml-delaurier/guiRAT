"""
RAT GUI (Retrieval Augmented Thinking) with CustomTkinter Interface
This module implements a GUI version of the RAT system, combining DeepSeek's reasoning 
capabilities with Groq's high-performance language models.
"""

import tkinter as tk
import customtkinter
from openai import OpenAI
from groq import Groq
import os
from dotenv import load_dotenv
import time
import threading
import json
import re

# Set appearance mode and default color theme
customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-green")  # Themes: blue (default), dark-blue, green

# Model Constants
REASONING_MODEL = "deepseek-r1-distill-llama-70b"

# Model Configurations
MODELS = {
    # All models available for both reasoning and base roles
    "reasoning": {
        "deepseek-r1-distill-llama-70b": {
            "name": "DeepSeek R1 Distill 70B",
            "type": "reasoning",
            "description": "Specialized model for generating detailed reasoning",
            "context_length": 128000
        },
        "gemma2-9b-it": {
            "name": "Gemma 2 9B IT",
            "type": "reasoning",
            "description": "Google's instruction-tuned 9B parameter model",
            "context_length": 8192
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B Versatile",
            "type": "reasoning",
            "description": "Meta's versatile large language model with extended context",
            "context_length": 128000
        },
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B Instant",
            "type": "reasoning",
            "description": "Fast and efficient 8B parameter model with extended context",
            "context_length": 128000
        },
        "llama-guard-3-8b": {
            "name": "Llama Guard 3 8B",
            "type": "reasoning",
            "description": "Meta's safety-focused model with 8B parameters",
            "context_length": 8192
        },
        "llama3-70b-8192": {
            "name": "Llama 3 70B",
            "type": "reasoning",
            "description": "Meta's powerful 70B parameter model",
            "context_length": 8192
        },
        "llama3-8b-8192": {
            "name": "Llama 3 8B",
            "type": "reasoning",
            "description": "Efficient 8B parameter model for general use",
            "context_length": 8192
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral 8x7B",
            "type": "reasoning",
            "description": "Mistral's mixture of experts model with extended context",
            "context_length": 32768
        },
        "llama-3.3-70b-specdec": {
            "name": "Llama 3.3 70B SpecDec",
            "type": "reasoning",
            "description": "Meta's specialized decoder variant of the 70B model",
            "context_length": 8192
        },
        "llama-3.2-1b-preview": {
            "name": "Llama 3.2 1B Preview",
            "type": "reasoning",
            "description": "Lightweight preview model with extended context",
            "context_length": 128000
        },
        "llama-3.2-3b-preview": {
            "name": "Llama 3.2 3B Preview",
            "type": "reasoning",
            "description": "Medium-sized preview model with extended context",
            "context_length": 128000
        },
        "llama-3.2-11b-vision-preview": {
            "name": "Llama 3.2 11B Vision Preview",
            "type": "reasoning",
            "description": "Vision-capable preview model with extended context",
            "context_length": 128000
        },
        "llama-3.2-90b-vision-preview": {
            "name": "Llama 3.2 90B Vision Preview",
            "type": "reasoning",
            "description": "Large vision-capable preview model with extended context",
            "context_length": 128000
        }
    },
    
    # Base Models - same models available here
    "base": {
        "deepseek-r1-distill-llama-70b": {
            "name": "DeepSeek R1 Distill 70B",
            "type": "base",
            "description": "Specialized model for generating detailed reasoning",
            "context_length": 128000
        },
        "gemma2-9b-it": {
            "name": "Gemma 2 9B IT",
            "type": "base",
            "description": "Google's instruction-tuned 9B parameter model",
            "context_length": 8192
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B Versatile",
            "type": "base",
            "description": "Meta's versatile large language model with extended context",
            "context_length": 128000
        },
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B Instant",
            "type": "base",
            "description": "Fast and efficient 8B parameter model with extended context",
            "context_length": 128000
        },
        "llama-guard-3-8b": {
            "name": "Llama Guard 3 8B",
            "type": "base",
            "description": "Meta's safety-focused model with 8B parameters",
            "context_length": 8192
        },
        "llama3-70b-8192": {
            "name": "Llama 3 70B",
            "type": "base",
            "description": "Meta's powerful 70B parameter model",
            "context_length": 8192
        },
        "llama3-8b-8192": {
            "name": "Llama 3 8B",
            "type": "base",
            "description": "Efficient 8B parameter model for general use",
            "context_length": 8192
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral 8x7B",
            "type": "base",
            "description": "Mistral's mixture of experts model with extended context",
            "context_length": 32768
        },
        "llama-3.3-70b-specdec": {
            "name": "Llama 3.3 70B SpecDec",
            "type": "base",
            "description": "Meta's specialized decoder variant of the 70B model",
            "context_length": 8192
        },
        "llama-3.2-1b-preview": {
            "name": "Llama 3.2 1B Preview",
            "type": "base",
            "description": "Lightweight preview model with extended context",
            "context_length": 128000
        },
        "llama-3.2-3b-preview": {
            "name": "Llama 3.2 3B Preview",
            "type": "base",
            "description": "Medium-sized preview model with extended context",
            "context_length": 128000
        },
        "llama-3.2-11b-vision-preview": {
            "name": "Llama 3.2 11B Vision Preview",
            "type": "base",
            "description": "Vision-capable preview model with extended context",
            "context_length": 128000
        },
        "llama-3.2-90b-vision-preview": {
            "name": "Llama 3.2 90B Vision Preview",
            "type": "base",
            "description": "Large vision-capable preview model with extended context",
            "context_length": 128000
        }
    }
}

# Default Models
DEFAULT_REASONING_MODEL = "deepseek-r1-distill-llama-70b"
DEFAULT_BASE_MODEL = "llama-3.3-70b-versatile"

class ModelChain:
    """
    ModelChain orchestrates a two-step reasoning process:
    1. Get reasoning from DeepSeek R1 model
    2. Use that reasoning to enhance responses from other models
    """

    def __init__(self):
        """Initialize the model chain with Groq client"""
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Message history tracking
        self.messages = []
        
        # Configuration settings
        self.current_reasoning_model = DEFAULT_REASONING_MODEL
        self.current_base_model = DEFAULT_BASE_MODEL
        self.show_reasoning = True
        
        # Default configuration for reasoning model
        self.reasoning_config = {
            "model": DEFAULT_REASONING_MODEL,
            "temperature": 0.6,
            "max_completion_tokens": 4096,
            "top_p": 0.95,
            "stream": True,
            "presence_penalty": 0.3,  # Slight penalty to reduce repetition
            "frequency_penalty": 0.4  # Encourage vocabulary diversity
        }
        
        # Default configuration for base model
        self.base_config = {
            "model": DEFAULT_BASE_MODEL,
            "temperature": 0.6,
            "max_completion_tokens": 4096,
            "top_p": 0.95,
            "stream": True,
            "presence_penalty": 0.2,  # Slight penalty to reduce repetition
            "frequency_penalty": 0.3  # Encourage vocabulary diversity
        }

    def extract_thinking(self, text):
        """
        Extract content between <think> tags and discard everything else.
        Returns only the clean content between tags, without the tags themselves.
        """
        import re
        
        # Use regex to find content between <think> tags
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows matching across newlines
        
        if match:
            # Return only the content inside the tags, stripped of leading/trailing whitespace
            return match.group(1).strip()
        
        # If no tags found, return empty string
        return ""

    def get_deepseek_reasoning(self, user_input):
        """
        Get reasoning from DeepSeek R1 model.
        
        Args:
            user_input (str): User's question or prompt
            
        Returns:
            str: Generated reasoning content
        """
        try:
            # Prepare the reasoning prompt
            messages = [{
                "role": "user",
                "content": f"Please think about how to best answer this question. Wrap your thinking in <think> tags.\nQuestion: {user_input}"
            }]
            
            # Get reasoning from DeepSeek R1
            completion = self.groq_client.chat.completions.create(
                model=self.current_reasoning_model,
                messages=messages,
                temperature=0.5,
                max_tokens=4096,
                top_p=0.95,
                stream=True
            )

            # Buffer for collecting streamed response
            response_buffer = ""
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_buffer += chunk.choices[0].delta.content
            
            # Extract only the content between <think> tags
            thinking_content = self.extract_thinking(response_buffer)
            
            # Return only the extracted thinking content
            yield {
                "prompt": messages[0]["content"],
                "response": thinking_content,  # Only the content between tags
                "type": "reasoning"
            }
            
        except Exception as e:
            yield {
                "prompt": messages[0]["content"],
                "response": f"Reasoning failed: {str(e)}",
                "type": "error"
            }

    def get_groq_response(self, reasoning):
        """
        Generate response using the reasoning to enhance the model's output.
        
        Args:
            reasoning (str): Generated reasoning from DeepSeek
            
        Returns:
            str: Generated response
        """
        try:
            # Create a prompt that focuses on generating a response based on the reasoning
            messages = [{
                "role": "system",
                "content": "You are a helpful AI assistant. Based on the following reasoning, provide a clear and direct response."
            }, {
                "role": "user",
                "content": reasoning
            }]
            
            # Get enhanced response using reasoning
            completion = self.groq_client.chat.completions.create(
                model=self.current_base_model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=4096,
                top_p=0.95,
                stream=True
            )

            # Buffer for collecting streamed response
            response_buffer = ""
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_buffer += chunk.choices[0].delta.content
            
            # Yield both the prompt and full response
            yield {
                "prompt": messages[1]["content"],
                "system_prompt": messages[0]["content"],
                "response": response_buffer,
                "type": "response"
            }
            
        except Exception as e:
            yield {
                "prompt": messages[1]["content"],
                "system_prompt": messages[0]["content"],
                "response": f"Response failed: {str(e)}",
                "type": "error"
            }

    def set_reasoning_model(self, model_name):
        """Set the current reasoning model"""
        if model_name in MODELS["reasoning"]:
            self.current_reasoning_model = model_name
            self.reasoning_config["model"] = model_name
            return True
        return False

    def set_base_model(self, model_name):
        """Set the current base model"""
        if model_name in MODELS["base"]:
            self.current_base_model = model_name
            self.base_config["model"] = model_name
            return True
        return False

class RatGUI(customtkinter.CTk):
    """Main GUI application for RAT"""
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("guiRAT - Retrieval Augmented Thinking")
        self.geometry("1200x800")
        
        # Set window icon if available
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        
        # Initialize model chain and state
        self.model_chain = ModelChain()
        self.running = False
        
        # Initialize logging
        self.log_file = "conversation_logs.txt"
        self.setup_logging()
        
        # Create GUI components
        self.create_widgets()
        
        # Configure window behavior
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_logging(self):
        """
        Set up the conversation logging system.
        Creates or appends to conversation_logs.txt with a session start marker.
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")

    def log_conversation(self, content, role, model_name=None, prompt=None, system_prompt=None):
        """
        Log conversation entries to the log file.
        
        Args:
            content (str): The message content
            role (str): The role (user/assistant)
            model_name (str, optional): The model name if applicable
            prompt (str, optional): The prompt sent to the model
            system_prompt (str, optional): The system prompt if applicable
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            if role == "user":
                f.write(f"[{timestamp}] USER: {content}\n\n")
            else:
                model_info = f" ({model_name})" if model_name else ""
                f.write(f"[{timestamp}] ASSISTANT{model_info}:\n")
                if system_prompt:
                    f.write(f"System Prompt: {system_prompt}\n")
                if prompt:
                    f.write(f"Input Prompt: {prompt}\n")
                f.write(f"Response: {content}\n\n")
                f.write("-" * 50 + "\n\n")

    def create_widgets(self):
        """Create all GUI components"""
        # Create main container
        self.main_container = customtkinter.CTkFrame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel at top
        self.create_control_panel(self.main_container)
        
        # Create output display in middle
        self.create_output_display(self.main_container)
        
        # Create input area at bottom
        self.create_input_area(self.main_container)

    def create_control_panel(self, parent):
        """Create the control panel with model selection and options"""
        # Main control frame
        control_frame = customtkinter.CTkFrame(parent)
        control_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Reasoning Model Selection
        reasoning_frame = customtkinter.CTkFrame(control_frame)
        reasoning_frame.pack(side=tk.LEFT, padx=(10, 10))
        
        customtkinter.CTkLabel(
            reasoning_frame, 
            text="Reasoning:",
            text_color="white"
        ).pack(side=tk.LEFT)
        
        self.reasoning_var = tk.StringVar(value=DEFAULT_REASONING_MODEL)
        reasoning_menu = customtkinter.CTkOptionMenu(
            reasoning_frame,
            values=[model for model in MODELS["reasoning"].keys()],
            variable=self.reasoning_var,
            command=self.change_reasoning_model,
            fg_color="#2b2b2b",
            button_color="#404040",
            button_hover_color="#505050",
            text_color="white",
            dynamic_resizing=False,
            width=150
        )
        reasoning_menu.pack(side=tk.LEFT, padx=(5, 0))
        
        # Base Model Selection
        base_frame = customtkinter.CTkFrame(control_frame)
        base_frame.pack(side=tk.LEFT, padx=10)
        
        customtkinter.CTkLabel(
            base_frame, 
            text="Base:",
            text_color="white"
        ).pack(side=tk.LEFT)
        
        self.base_var = tk.StringVar(value=DEFAULT_BASE_MODEL)
        base_menu = customtkinter.CTkOptionMenu(
            base_frame,
            values=[model for model in MODELS["base"].keys()],
            variable=self.base_var,
            command=self.change_base_model,
            fg_color="#2b2b2b",
            button_color="#404040",
            button_hover_color="#505050",
            text_color="white",
            dynamic_resizing=False,
            width=150
        )
        base_menu.pack(side=tk.LEFT, padx=(5, 0))

        # Groq API Key Input
        api_frame = customtkinter.CTkFrame(control_frame)
        api_frame.pack(side=tk.LEFT, padx=10)
        
        customtkinter.CTkLabel(
            api_frame, 
            text="Groq API Key:",
            text_color="white"
        ).pack(side=tk.LEFT)
        
        # Get API key from environment
        initial_api_key = os.getenv("GROQ_API_KEY", "")
        self.api_key_var = tk.StringVar(value=initial_api_key)
        
        # API Key entry with password masking
        self.api_key_entry = customtkinter.CTkEntry(
            api_frame,
            textvariable=self.api_key_var,
            width=150,
            show="•",  # Mask the API key
            fg_color="#2b2b2b",
            text_color="white",
            placeholder_text="Enter Groq API Key"
        )
        self.api_key_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Save button
        self.save_api_button = customtkinter.CTkButton(
            api_frame,
            text="Save",
            command=self.save_api_key,
            fg_color="#2b2b2b",
            hover_color="#404040",
            corner_radius=5,
            border_width=1,
            border_color="#404040",
            width=50
        )
        self.save_api_button.pack(side=tk.LEFT, padx=(5, 0))

    def create_output_display(self, parent):
        """Create the output text display area with bubble styling"""
        # Create a frame to hold the output
        self.output_frame = customtkinter.CTkFrame(parent)
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Output text widget with custom styling
        self.output_text = customtkinter.CTkTextbox(
            self.output_frame,
            width=1000,
            height=500,
            wrap=tk.WORD,
            text_color="white",
            corner_radius=5,
            fg_color="#1e1e1e"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Store collapsed state for reasoning bubbles
        self.collapsed_bubbles = {}
        self.next_bubble_id = 0
        
        # Configure text tags for different types of output
        self.output_text._textbox.tag_config(
            'user_bubble', 
            background="#2d2d2d",
            foreground="white",
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text._textbox.tag_config(
            'reasoning_bubble',
            background="#1e3d59",  # Dark blue
            foreground="#b3e5fc",  # Light blue
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text._textbox.tag_config(
            'response_bubble',
            background="#1b5e20",  # Dark green
            foreground="#b9f6ca",  # Light green
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text._textbox.tag_config(
            'bubble_header',
            foreground="#ffd700",  # Gold
            font=('Consolas', 10, 'bold'),
            spacing1=5,
            lmargin1=10,
            lmargin2=10
        )
        
        self.output_text._textbox.tag_config(
            'bubble_metadata',
            foreground="#ffd700",  # Gold
            font=('Consolas', 9),
            spacing3=5,
            lmargin1=20,
            lmargin2=20
        )
        
        self.output_text._textbox.tag_config(
            'error_bubble',
            background="#b71c1c",  # Dark red
            foreground="#ffcdd2",  # Light red
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text._textbox.tag_config(
            'clickable_header',
            font=('Consolas', 10, 'bold'),
            foreground="#ffd700",  # Gold
            spacing1=5,
            lmargin1=10,
            lmargin2=10
        )
        
        # Configure tag bindings for cursor changes
        self.output_text._textbox.tag_bind('clickable_header', '<Enter>', 
            lambda e: self.output_text._textbox.configure(cursor="hand2"))
        self.output_text._textbox.tag_bind('clickable_header', '<Leave>', 
            lambda e: self.output_text._textbox.configure(cursor="arrow"))
        
        # Add click handler for collapse/expand
        self.output_text._textbox.tag_bind('collapse_button', '<Button-1>', self.toggle_reasoning_bubble)
        
        # Make output read-only
        self.output_text.configure(state='disabled')

    def format_bubble_header(self, model_name, tokens, time_taken, bubble_id=None, is_reasoning=False):
        """Format the bubble header with model info and collapse button if needed"""
        model_info = MODELS['reasoning' if 'deepseek' in model_name else 'base'][model_name]
        
        if is_reasoning:
            collapse_emoji = "▼" if not self.collapsed_bubbles.get(bubble_id, True) else "▶"
            return (
                f"┌─ {model_info['name']} {collapse_emoji}\n"
                f"└─ {tokens} tokens | {time_taken:.2f}s"
            )
        else:
            return (
                f"┌─ {model_info['name']}\n"
                f"└─ {tokens} tokens | {time_taken:.2f}s"
            )

    def toggle_reasoning_bubble(self, event):
        """Toggle the visibility of a reasoning bubble's content"""
        # Get the index of the clicked position
        clicked_index = self.output_text._textbox.index(f"@{event.x},{event.y}")
        
        # Find all tags at this position
        tags = self.output_text._textbox.tag_names(clicked_index)
        
        for tag in tags:
            if tag.startswith('bubble_header_'):
                bubble_id = int(tag.split('_')[2])
                
                self.output_text.configure(state='normal')
                
                # Toggle collapsed state
                is_collapsed = not self.collapsed_bubbles.get(bubble_id, True)
                self.collapsed_bubbles[bubble_id] = is_collapsed
                
                # Get the entire header text
                header_start = self.output_text._textbox.index(f"{tag}.first")
                header_end = self.output_text._textbox.index(f"{tag}.last")
                header_text = self.output_text._textbox.get(header_start, header_end)
                
                # Replace the arrow emoji
                new_header = header_text.replace("▼", "▶") if is_collapsed else header_text.replace("▶", "▼")
                self.output_text._textbox.delete(header_start, header_end)
                self.output_text._textbox.insert(header_start, new_header, tag)
                
                # Toggle content visibility
                content_tag = f'bubble_content_{bubble_id}'
                if is_collapsed:
                    self.output_text._textbox.tag_config(content_tag, elide=True)
                else:
                    self.output_text._textbox.tag_config(content_tag, elide=False)
                
                self.output_text.configure(state='disabled')
                break

    def append_bubble(self, content, tag, model_name=None, tokens=0, time_taken=0):
        """Append a bubble with labeled frame and metadata"""
        self.output_text.configure(state='normal')
        
        # Add newline before bubble
        self.output_text._textbox.insert(tk.END, "\n")
        
        # Check if this is a reasoning bubble
        is_reasoning = tag == 'reasoning_bubble'
        bubble_id = None
        
        # Add bubble header if model info is provided
        if model_name:
            if is_reasoning:
                bubble_id = self.next_bubble_id
                self.next_bubble_id += 1
                header_tag = f'bubble_header_{bubble_id}'
                
                # Create header with clickable arrow
                header = self.format_bubble_header(model_name, tokens, time_taken, bubble_id, is_reasoning)
                
                # Insert header and make it clickable
                self.output_text._textbox.insert(tk.END, header + "\n", (header_tag, 'clickable_header'))
                
                # Bind click event to the header
                self.output_text._textbox.tag_bind(header_tag, '<Button-1>', self.toggle_reasoning_bubble)
            else:
                header = self.format_bubble_header(model_name, tokens, time_taken)
                self.output_text._textbox.insert(tk.END, header + "\n", 'bubble_header')
        
        # Add the main content in a bubble
        content_tag = f'bubble_content_{bubble_id}' if bubble_id is not None else tag
        self.output_text._textbox.insert(tk.END, content + "\n", (content_tag, tag))
        
        # Set initial state for reasoning bubbles (collapsed by default)
        if is_reasoning:
            self.collapsed_bubbles[bubble_id] = True
            self.output_text._textbox.tag_config(content_tag, elide=True)
        
        self.output_text.configure(state='disabled')
        self.output_text._textbox.see(tk.END)
        self.update_idletasks()

    def create_input_area(self, parent):
        """Create the input area with entry and submit button"""
        # Main input frame
        input_frame = customtkinter.CTkFrame(parent)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Create and configure the input text widget
        self.input_text = customtkinter.CTkTextbox(
            input_frame,
            width=1000,
            height=100,
            wrap=tk.WORD,
            text_color="white",
            corner_radius=5,
            fg_color="#363636"
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Bind Return key to process input
        self.input_text._textbox.bind('<Return>', lambda e: self.process_input())
        
        # Button frame for Send and Clear buttons
        button_frame = customtkinter.CTkFrame(input_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Send button with hover effects
        self.send_button = customtkinter.CTkButton(
            button_frame,
            text="Send",
            command=self.process_input,
            fg_color="#2b2b2b",
            hover_color="#404040",
            corner_radius=5,
            border_width=1,
            border_color="#404040",
            width=50
        )
        self.send_button.pack(side=tk.TOP, pady=(0, 5))
        
        # Clear button with hover effects
        self.clear_button = customtkinter.CTkButton(
            button_frame,
            text="Clear",
            command=self.clear_output,
            fg_color="#2b2b2b",
            hover_color="#404040",
            corner_radius=5,
            border_width=1,
            border_color="#404040"
        )
        self.clear_button.pack(side=tk.TOP)

    def process_input(self, event=None):
        """Handle user input submission"""
        if self.running:
            return
            
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return
            
        self.input_text.delete("1.0", tk.END)
        self.append_bubble(
            f"You: {user_input}",
            'user_bubble'
        )
        self.log_conversation(user_input, "user")
        
        # Start processing thread
        self.running = True
        threading.Thread(
            target=self.handle_query,
            args=(user_input,),
            daemon=True
        ).start()

    def handle_query(self, user_input):
        """Process the query in a separate thread"""
        try:
            # Log user input
            self.log_conversation(user_input, "user")
            
            # Get reasoning from DeepSeek
            reasoning_start = time.time()  # Initialize timing
            async_gen = self.model_chain.get_deepseek_reasoning(user_input)
            reasoning_content = ""
            
            for result in async_gen:
                if result["type"] == "reasoning":
                    reasoning_content = result["response"]  # This is already clean, between-tags content
                    self.append_bubble(
                        reasoning_content,  # Show only the clean content
                        'reasoning_bubble',
                        self.model_chain.current_reasoning_model,
                        len(reasoning_content.split()),
                        time.time() - reasoning_start
                    )
                    # Log only the clean content
                    self.log_conversation(
                        reasoning_content, 
                        "assistant",
                        self.model_chain.current_reasoning_model,
                        prompt=result["prompt"]
                    )
                elif result["type"] == "error":
                    self.append_bubble(result["response"], "error")
                    self.log_conversation(
                        result["response"],
                        "assistant",
                        "ERROR",
                        prompt=result["prompt"]
                    )
                    return
            
            # Get enhanced response using the clean reasoning text
            response_start = time.time()
            if reasoning_content:  # Only proceed if we have thinking content
                response_generator = self.model_chain.get_groq_response(reasoning_content)
                
                for result in response_generator:
                    if result["type"] == "response":
                        self.append_bubble(
                            result["response"],
                            'response_bubble',
                            self.model_chain.current_base_model,
                            len(result["response"].split()),
                            time.time() - response_start
                        )
                        self.log_conversation(
                            result["response"],
                            "assistant",
                            self.model_chain.current_base_model,
                            prompt=reasoning_content,  # Using clean content as prompt
                            system_prompt=result["system_prompt"]
                        )
                    elif result["type"] == "error":
                        self.append_bubble(result["response"], "error")
                        self.log_conversation(
                            result["response"],
                            "assistant",
                            "ERROR",
                            prompt=reasoning_content,
                            system_prompt=result["system_prompt"]
                        )
            else:
                error_msg = "Error: No thinking content found in model response"
                self.append_bubble(error_msg, "error")
                self.log_conversation(error_msg, "assistant", "ERROR")
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.append_bubble(error_msg, "error")
            self.log_conversation(error_msg, "assistant", "ERROR")
        finally:
            self.running = False

    def change_reasoning_model(self, model_name):
        """Handle reasoning model change"""
        if self.model_chain.set_reasoning_model(model_name):
            self.append_bubble(
                f"Switched reasoning model to: {MODELS['reasoning'][model_name]['name']}",
                'metadata'
            )
        else:
            self.show_error("Invalid reasoning model selection")
            self.reasoning_var.set(DEFAULT_REASONING_MODEL)

    def change_base_model(self, model_name):
        """Handle base model change"""
        if self.model_chain.set_base_model(model_name):
            self.append_bubble(
                f"Switched base model to: {MODELS['base'][model_name]['name']}",
                'metadata'
            )
        else:
            self.show_error("Invalid base model selection")
            self.base_var.set(DEFAULT_BASE_MODEL)

    def clear_output(self):
        """Clear the output display"""
        self.output_text.configure(state='normal')
        self.output_text._textbox.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')
        self.model_chain.messages = []

    def show_error(self, message):
        """Show an error message in a CustomTkinter dialog"""
        dialog = customtkinter.CTkToplevel(self)
        dialog.title("Error")
        dialog.geometry("400x150")
        
        # Make dialog modal
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Add error message
        label = customtkinter.CTkLabel(
            dialog,
            text=message,
            text_color="white",
            wraplength=350
        )
        label.pack(pady=20)
        
        # Add OK button
        button = customtkinter.CTkButton(
            dialog,
            text="OK",
            command=dialog.destroy,
            fg_color="#2b2b2b",
            hover_color="#404040"
        )
        button.pack(pady=10)

    def save_api_key(self):
        """Save the API key to both environment and .env file"""
        api_key = self.api_key_var.get().strip()
        
        if not api_key:
            self.show_error("Please enter a valid API key")
            return
        
        # Update environment variable
        os.environ["GROQ_API_KEY"] = api_key
        
        # Update .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        env_content = ""
        
        # Read existing content if file exists
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if not line.startswith('GROQ_API_KEY='):
                        env_content += line
        
        # Add new API key
        env_content += f"\nGROQ_API_KEY={api_key}\n"
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            f.write(env_content.strip())
        
        # Update the Groq client
        self.model_chain.groq_client = Groq(api_key=api_key)
        
        # Show success message in chat
        self.append_bubble(
            "Groq API key has been updated and saved",
            'metadata'
        )

    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.destroy()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Create and run application
    app = RatGUI()
    app.mainloop()
