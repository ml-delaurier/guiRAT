"""
RAT GUI (Retrieval Augmented Thinking) with Tkinter Interface
This module implements a GUI version of the RAT system, combining DeepSeek's reasoning 
capabilities with Groq's high-performance language models.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from openai import OpenAI
from groq import Groq
import os
from dotenv import load_dotenv
import time
import threading
import json
import re

# Model Constants
REASONING_MODEL = "deepseek-r1-distill-llama-70b"

# Model Configurations
MODELS = {
    # Reasoning Model
    "reasoning": {
        "deepseek-r1-distill-llama-70b": {
            "name": "DeepSeek R1 Distill 70B",
            "type": "reasoning",
            "description": "Specialized model for generating detailed reasoning",
            "context_length": 8192
        }
    },
    
    # Base Models
    "base": {
        "llama-3.3-70b-specdec": {
            "name": "Llama 3.3 70B SpecDec",
            "type": "base",
            "description": "Meta's Llama model optimized for specialized tasks",
            "context_length": 8192
        },
        "llama-3.2-1b-preview": {
            "name": "Llama 3.2 1B Preview",
            "type": "base",
            "description": "Lightweight preview model with 128k context window",
            "context_length": 8192
        },
        "llama-3.2-3b-preview": {
            "name": "Llama 3.2 3B Preview",
            "type": "base",
            "description": "Medium-sized preview model with 128k context window",
            "context_length": 8192
        },
        "llama-3.2-11b-vision-preview": {
            "name": "Llama 3.2 11B Vision Preview",
            "type": "base",
            "description": "Vision-capable preview model with 128k context window",
            "context_length": 8192
        },
        "llama-3.2-90b-vision-preview": {
            "name": "Llama 3.2 90B Vision Preview",
            "type": "base",
            "description": "Large vision-capable preview model with 128k context window",
            "context_length": 8192
        }
    }
}

# Default Models
DEFAULT_REASONING_MODEL = REASONING_MODEL
DEFAULT_BASE_MODEL = "llama-3.3-70b-specdec"

# Reasoning Format Options
REASONING_FORMATS = {
    "parsed": "Separates reasoning into a dedicated field while keeping the response concise.",
    "raw": "Includes reasoning within think tags in the content.",
    "hidden": "Returns only the final answer for maximum efficiency."
}

# Default Model Configuration
DEFAULT_CONFIG = {
    "temperature": 0.6,        # Controls randomness (0.5-0.7 recommended)
    "max_tokens": 4096,        # Increased for complex reasoning
    "top_p": 0.95,            # Controls diversity of token selection
    "stream": True,           # Enable streaming for interactive tasks
    "stop": None,             # No custom stop sequences
    "seed": None,             # For reproducible results if needed
    "json_mode": False,       # Enable for structured output
    "presence_penalty": 0.3,  # Slight penalty to reduce repetition
    "frequency_penalty": 0.4  # Encourage vocabulary diversity
}

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
        self.reasoning_format = "raw"
        
        # Default configuration for reasoning model
        self.reasoning_config = {
            "model": DEFAULT_REASONING_MODEL,
            "temperature": 0.6,
            "max_completion_tokens": 4096,
            "top_p": 0.95,
            "stream": True,
            "presence_penalty": 0.3,  # Slight penalty to reduce repetition
            "frequency_penalty": 0.4,  # Encourage vocabulary diversity
            "reasoning_format": "raw"
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

    def set_model(self, model_name):
        """Set the current model"""
        if model_name in MODELS["reasoning"]:
            self.current_reasoning_model = model_name
            self.reasoning_config["model"] = model_name
            return True
        elif model_name in MODELS["base"]:
            self.current_base_model = model_name
            self.base_config["model"] = model_name
            return True
        return False

    def set_reasoning_format(self, format_name):
        """Set the reasoning format"""
        if format_name in REASONING_FORMATS:
            self.reasoning_format = format_name
            self.reasoning_config["reasoning_format"] = format_name
            return True
        return False

    def get_model_display_name(self):
        """Get the display name of the current model"""
        if self.current_reasoning_model in MODELS["reasoning"]:
            return MODELS["reasoning"][self.current_reasoning_model]["name"]
        elif self.current_base_model in MODELS["base"]:
            return MODELS["base"][self.current_base_model]["name"]
        return "Unknown Model"

class RatGUI(tk.Tk):
    """Main GUI application for RAT"""
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("guiRAT - Retrieval Augmented Thinking")
        self.geometry("1200x800")
        self.configure(bg='#2b2b2b')  # Dark theme background
        
        # Initialize model chain and state
        self.model_chain = ModelChain()
        self.running = False
        
        # Initialize logging
        self.log_file = "conversation_full.txt"
        self.setup_logging()
        
        # Create GUI components
        self.create_widgets()
        
        # Configure window behavior
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_logging(self):
        """
        Set up the conversation logging system.
        Creates or appends to conversation_full.txt with a session start marker.
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
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#2b2b2b")
        style.configure("Custom.TButton", 
                       padding=6, 
                       background="#3c3f41",
                       foreground="white")
        style.configure("Custom.TLabel", 
                       background="#2b2b2b",
                       foreground="white")


        main_frame = ttk.Frame(self, style="Custom.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel at top
        self.create_control_panel(main_frame)
        
        # Output display
        self.create_output_display(main_frame)
        
        # Input area at bottom
        self.create_input_area(main_frame)

    def create_control_panel(self, parent):
        """Create the control panel with model selection and options"""
        control_frame = ttk.Frame(parent, style="Custom.TFrame")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Reasoning Model Selection
        reasoning_frame = ttk.Frame(control_frame, style="Custom.TFrame")
        reasoning_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(reasoning_frame, 
                 text="Reasoning Model:", 
                 style="Custom.TLabel").pack(side=tk.LEFT)
        
        self.reasoning_var = tk.StringVar(value=DEFAULT_REASONING_MODEL)
        reasoning_menu = ttk.OptionMenu(
            reasoning_frame,
            self.reasoning_var,
            DEFAULT_REASONING_MODEL,
            *[model for model in MODELS["reasoning"].keys()],
            command=self.change_reasoning_model
        )
        reasoning_menu.pack(side=tk.LEFT, padx=(5, 0))
        
        # Base Model Selection
        base_frame = ttk.Frame(control_frame, style="Custom.TFrame")
        base_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(base_frame, 
                 text="Base Model:", 
                 style="Custom.TLabel").pack(side=tk.LEFT)
        
        self.base_var = tk.StringVar(value=DEFAULT_BASE_MODEL)
        base_menu = ttk.OptionMenu(
            base_frame,
            self.base_var,
            DEFAULT_BASE_MODEL,
            *[model for model in MODELS["base"].keys()],
            command=self.change_base_model
        )
        base_menu.pack(side=tk.LEFT, padx=(5, 0))
        
        # Add tooltips for model descriptions
        self.create_model_tooltips(reasoning_menu, base_menu)
        
        # Reasoning format selection
        format_frame = ttk.Frame(control_frame, style="Custom.TFrame")
        format_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(format_frame, 
                 text="Format:", 
                 style="Custom.TLabel").pack(side=tk.LEFT)
        
        self.format_var = tk.StringVar(value='raw')
        format_menu = ttk.OptionMenu(
            format_frame,
            self.format_var,
            'raw',
            *REASONING_FORMATS.keys(),
            command=self.change_format
        )
        format_menu.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create tooltip for format menu
        self.tooltip = None
        format_menu.bind('<Enter>', self.show_format_tooltip)
        format_menu.bind('<Leave>', self.hide_format_tooltip)
        
        # Reasoning visibility toggle
        self.reasoning_visibility_var = tk.BooleanVar(value=True)
        reasoning_btn = ttk.Checkbutton(
            control_frame,
            text="Show Reasoning",
            variable=self.reasoning_visibility_var,
            command=self.toggle_reasoning,
            style="Custom.TCheckbutton"
        )
        reasoning_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_btn = ttk.Button(
            control_frame,
            text="Clear",
            command=self.clear_output,
            style="Custom.TButton"
        )
        clear_btn.pack(side=tk.RIGHT)

    def create_model_tooltips(self, reasoning_menu, base_menu):
        """Create tooltips for model descriptions"""
        def create_tooltip(menu, model_type):
            def show_tooltip(event):
                model_var = self.reasoning_var if model_type == "reasoning" else self.base_var
                model_name = model_var.get()
                description = MODELS[model_type][model_name]["description"]
                
                x = event.x_root + 20
                y = event.y_root + 10
                
                if not hasattr(self, f'{model_type}_tooltip'):
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    label = ttk.Label(
                        tooltip,
                        text=description,
                        justify=tk.LEFT,
                        relief=tk.SOLID,
                        borderwidth=1,
                        background="#2b2b2b",
                        foreground="white",
                        padding=(5, 2)
                    )
                    label.pack()
                    setattr(self, f'{model_type}_tooltip', tooltip)
                    setattr(self, f'{model_type}_tooltip_label', label)
                else:
                    tooltip = getattr(self, f'{model_type}_tooltip')
                    label = getattr(self, f'{model_type}_tooltip_label')
                    label.configure(text=description)
                
                tooltip.geometry(f"+{x}+{y}")
                tooltip.deiconify()
                tooltip.lift()
                tooltip.attributes('-topmost', True)
            
            def hide_tooltip(event):
                if hasattr(self, f'{model_type}_tooltip'):
                    getattr(self, f'{model_type}_tooltip').withdraw()
            
            menu.bind('<Enter>', show_tooltip)
            menu.bind('<Leave>', hide_tooltip)
        
        create_tooltip(reasoning_menu, "reasoning")
        create_tooltip(base_menu, "base")

    def show_format_tooltip(self, event):
        """Show tooltip with format description"""
        current_format = self.format_var.get()
        description = REASONING_FORMATS.get(current_format, "")
        
        x = event.x_root + 20
        y = event.y_root + 10
        
        if not self.tooltip:
            self.tooltip = tk.Toplevel()
            self.tooltip.wm_overrideredirect(True)
            self.tooltip_label = ttk.Label(
                self.tooltip,
                text=description,
                justify=tk.LEFT,
                relief=tk.SOLID,
                borderwidth=1,
                background="#2b2b2b",
                foreground="white",
                padding=(5, 2)
            )
            self.tooltip_label.pack()
        else:
            self.tooltip_label.configure(text=description)
        
        self.tooltip.geometry(f"+{x}+{y}")
        self.tooltip.deiconify()
        self.tooltip.lift()
        self.tooltip.attributes('-topmost', True)

    def hide_format_tooltip(self, event):
        """Hide the tooltip"""
        if self.tooltip:
            self.tooltip.withdraw()

    def create_output_display(self, parent):
        """Create the output text display area with bubble styling"""
        # Create a frame to hold the output
        self.output_frame = ttk.Frame(parent, style="Custom.TFrame")
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.output_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Output text widget with custom styling
        self.output_text = tk.Text(
            self.output_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg='#1e1e1e',
            fg='white',
            insertbackground='white',
            height=20,
            padx=10,
            pady=10,
            yscrollcommand=scrollbar.set,
            cursor="arrow"  # Default cursor
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.output_text.yview)
        
        # Store collapsed state for reasoning bubbles
        self.collapsed_bubbles = {}
        self.next_bubble_id = 0
        
        # Configure text tags for different types of output
        self.output_text.tag_config(
            'user_bubble', 
            background='#2d2d2d',
            foreground='white',
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text.tag_config(
            'reasoning_bubble',
            background='#1e3d59',  # Dark blue
            foreground='#b3e5fc',  # Light blue
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text.tag_config(
            'response_bubble',
            background='#1b5e20',  # Dark green
            foreground='#b9f6ca',  # Light green
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text.tag_config(
            'bubble_header',
            foreground='#ffd700',  # Gold
            font=('Consolas', 10, 'bold'),
            spacing1=5,
            lmargin1=10,
            lmargin2=10
        )
        
        self.output_text.tag_config(
            'bubble_metadata',
            foreground='#ffd700',  # Gray
            font=('Consolas', 9),
            spacing3=5,
            lmargin1=20,
            lmargin2=20
        )
        
        self.output_text.tag_config(
            'error_bubble',
            background='#b71c1c',  # Dark red
            foreground='#ffcdd2',  # Light red
            spacing1=10,
            spacing3=10,
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
        
        self.output_text.tag_config(
            'clickable_header',
            font=('Consolas', 10, 'bold'),
            foreground='#ffd700',  # Gold
            spacing1=5,
            lmargin1=10,
            lmargin2=10
        )
        
        # Configure tag bindings for cursor changes
        self.output_text.tag_bind('clickable_header', '<Enter>', 
            lambda e: self.output_text.config(cursor="hand2"))  # Change to hand cursor on hover
        self.output_text.tag_bind('clickable_header', '<Leave>', 
            lambda e: self.output_text.config(cursor="arrow"))  # Change back to arrow cursor
        
        # Add click handler for collapse/expand
        self.output_text.tag_bind('collapse_button', '<Button-1>', self.toggle_reasoning_bubble)
        
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
        clicked_index = self.output_text.index(f"@{event.x},{event.y}")
        
        # Find all tags at this position
        tags = self.output_text.tag_names(clicked_index)
        
        for tag in tags:
            if tag.startswith('bubble_header_'):
                bubble_id = int(tag.split('_')[2])
                
                self.output_text.configure(state='normal')
                
                # Toggle collapsed state
                is_collapsed = not self.collapsed_bubbles.get(bubble_id, True)
                self.collapsed_bubbles[bubble_id] = is_collapsed
                
                # Get the entire header text
                header_start = self.output_text.index(f"{tag}.first")
                header_end = self.output_text.index(f"{tag}.last")
                header_text = self.output_text.get(header_start, header_end)
                
                # Replace the arrow emoji
                new_header = header_text.replace("▼", "▶") if is_collapsed else header_text.replace("▶", "▼")
                self.output_text.delete(header_start, header_end)
                self.output_text.insert(header_start, new_header, tag)
                
                # Toggle content visibility
                content_tag = f'bubble_content_{bubble_id}'
                if is_collapsed:
                    self.output_text.tag_config(content_tag, elide=True)
                else:
                    self.output_text.tag_config(content_tag, elide=False)
                
                self.output_text.configure(state='disabled')
                break

    def append_bubble(self, content, tag, model_name=None, tokens=0, time_taken=0):
        """Append a bubble with labeled frame and metadata"""
        self.output_text.configure(state='normal')
        
        # Add newline before bubble
        self.output_text.insert(tk.END, "\n")
        
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
                self.output_text.insert(tk.END, header + "\n", (header_tag, 'clickable_header'))
                
                # Bind click event to the header
                self.output_text.tag_bind(header_tag, '<Button-1>', self.toggle_reasoning_bubble)
            else:
                header = self.format_bubble_header(model_name, tokens, time_taken)
                self.output_text.insert(tk.END, header + "\n", 'bubble_header')
        
        # Add the main content in a bubble
        content_tag = f'bubble_content_{bubble_id}' if bubble_id is not None else tag
        self.output_text.insert(tk.END, content + "\n", (content_tag, tag))
        
        # Set initial state for reasoning bubbles (collapsed by default)
        if is_reasoning:
            self.collapsed_bubbles[bubble_id] = True
            self.output_text.tag_config(content_tag, elide=True)
        
        self.output_text.configure(state='disabled')
        self.output_text.see(tk.END)
        self.update_idletasks()

    def create_input_area(self, parent):
        """Create the input area with entry and submit button"""
        input_frame = ttk.Frame(parent, style="Custom.TFrame")
        input_frame.pack(fill=tk.X)
        
        # Input entry with modern styling
        self.input_entry = ttk.Entry(
            input_frame,
            font=('Consolas', 12),
            style="Custom.TEntry"
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.process_input)
        
        # Submit button
        submit_btn = ttk.Button(
            input_frame,
            text="Send",
            command=lambda: self.process_input(None),
            style="Custom.TButton"
        )
        submit_btn.pack(side=tk.LEFT)

    def process_input(self, event=None):
        """Handle user input submission"""
        if self.running:
            return
            
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
            
        self.input_entry.delete(0, tk.END)
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
                    if self.model_chain.show_reasoning:
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
        if model_name in MODELS["reasoning"]:
            if self.model_chain.set_model(model_name):
                self.append_bubble(
                    f"\nReasoning model changed to: {MODELS['reasoning'][model_name]['name']}\n",
                    'metadata'
                )
            else:
                messagebox.showerror("Error", "Invalid reasoning model selection")
                self.reasoning_var.set(DEFAULT_REASONING_MODEL)

    def change_base_model(self, model_name):
        """Handle base model change"""
        if model_name in MODELS["base"]:
            if self.model_chain.set_model(model_name):
                self.append_bubble(
                    f"\nBase model changed to: {MODELS['base'][model_name]['name']}\n",
                    'metadata'
                )
            else:
                messagebox.showerror("Error", "Invalid base model selection")
                self.base_var.set(DEFAULT_BASE_MODEL)

    def change_format(self, format_name):
        """Handle format change"""
        if self.model_chain.set_reasoning_format(format_name):
            self.append_bubble(
                f"\nFormat changed to: {format_name}\n",
                'metadata'
            )
        else:
            messagebox.showerror("Error", "Invalid format selection")

    def toggle_reasoning(self):
        """Toggle reasoning visibility"""
        self.model_chain.show_reasoning = self.reasoning_visibility_var.get()
        status = "visible" if self.model_chain.show_reasoning else "hidden"
        self.append_bubble(
            f"\nReasoning {status}\n",
            'metadata'
        )

    def clear_output(self):
        """Clear the output display"""
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')
        self.model_chain.messages = []

    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.quit()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Create and run application
    app = RatGUI()
    app.mainloop()
