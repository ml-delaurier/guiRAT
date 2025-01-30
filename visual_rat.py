import math
import threading
import time
from tkinter import ttk

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
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

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
    """Main GUI application for RAT - Visual Thinking Interface"""
    def __init__(self):
        super().__init__()
        self.title("VisualRAT - Cognitive Process Visualizer")
        self.geometry("1400x900")
        
        # Initialize model chain and state
        self.model_chain = ModelChain()
        self.running = False
        self.thinking_graph = {}  # Store reasoning components
        self.current_view = "Radial"
        self.node_size = 80
        
        # Add conversation history tracking
        self.conversation_history = []  # Store all conversation turns
        self.last_node_position = {"x": 400, "y": 300}  # Track last node position for continuous layout
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create GUI components
        self.create_sidebar()
        self.create_main_panels()
        self.create_input_panel()
        
        # Animation state
        self.loading_active = False
        self.animation_angle = 0

    def create_sidebar(self):
        """Create left sidebar with model controls"""
        self.sidebar = customtkinter.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        # Model selection
        customtkinter.CTkLabel(self.sidebar, text="Reasoning Model:").pack(pady=(10,0))
        self.reasoning_dropdown = customtkinter.CTkOptionMenu(
            self.sidebar, 
            values=list(MODELS["reasoning"].keys()),
            command=self.change_reasoning_model
        )
        self.reasoning_dropdown.pack(pady=5)
        
        customtkinter.CTkLabel(self.sidebar, text="Base Model:").pack(pady=(10,0))
        self.base_dropdown = customtkinter.CTkOptionMenu(
            self.sidebar, 
            values=list(MODELS["base"].keys()),
            command=self.change_base_model
        )
        self.base_dropdown.pack(pady=5)
        
        # Visualization controls
        customtkinter.CTkLabel(self.sidebar, text="View Mode:").pack(pady=(20,0))
        self.view_mode = customtkinter.CTkSegmentedButton(
            self.sidebar,
            values=["Flow", "Tree", "Radial"],
            command=self.change_view_mode
        )
        self.view_mode.set("Radial")
        self.view_mode.pack(pady=5)
        
        # Visualization settings
        customtkinter.CTkLabel(self.sidebar, text="Node Size:").pack(pady=(20,0))
        self.node_slider = customtkinter.CTkSlider(
            self.sidebar,
            from_=50,
            to=150,
            command=self.update_node_size
        )
        self.node_slider.set(80)
        self.node_slider.pack(pady=5)

    def create_main_panels(self):
        """Create main visualization and response panels"""
        main_container = customtkinter.CTkFrame(self)
        main_container.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        main_container.grid_columnconfigure(0, weight=3)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Visualization Canvas
        self.canvas = tk.Canvas(main_container, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        
        # Response Panel
        self.response_panel = customtkinter.CTkTextbox(
            main_container, 
            wrap="word",
            font=("Consolas", 12),
            fg_color="#1e1e1e",
            text_color="white"
        )
        self.response_panel.grid(row=0, column=1, sticky="nsew")

    def create_input_panel(self):
        """Create bottom input panel"""
        input_frame = customtkinter.CTkFrame(self)
        input_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_entry = customtkinter.CTkEntry(
            input_frame, 
            placeholder_text="Enter your query...",
            height=40
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self.input_entry.bind("<Return>", self.process_input)
        
        self.send_button = customtkinter.CTkButton(
            input_frame, 
            text="Analyze",
            command=self.process_input,
            width=100
        )
        self.send_button.grid(row=0, column=1, sticky="e")

    def visualize_thought_process(self, reasoning_steps):
        """Visualize the reasoning process on canvas"""
        self.canvas.delete("all")
        if not reasoning_steps:
            return

        if self.current_view == "Radial":
            self.draw_radial_layout(reasoning_steps, len(self.conversation_history))
        elif self.current_view == "Flow":
            self.draw_flow_layout(reasoning_steps, len(self.conversation_history))
        elif self.current_view == "Tree":
            self.draw_tree_layout(reasoning_steps, len(self.conversation_history))

    def draw_radial_layout(self, steps, conversation_turn):
        """Radial layout with central concept and connected nodes"""
        # Calculate center position based on conversation turn
        center_x = 400 + (conversation_turn * 50)  # Shift right for each turn
        center_y = 300
        
        # Create central node
        main_node = steps[0] if steps else "Query"
        self.create_node(center_x, center_y, main_node[:20], "#2FA4FF", main_node)
        
        # Position other nodes in a circle around the center
        num_nodes = len(steps) - 1
        if num_nodes > 0:
            radius = max(150, self.node_size * 2)
            for i, step in enumerate(steps[1:], 1):
                angle = (2 * math.pi * i) / num_nodes - math.pi/2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                node = self.create_node(x, y, step[:20], "#FF6B6B", step)
                
                # Draw connection line
                self.canvas.create_line(center_x, center_y, x, y, fill="#555555", width=2)
        
        # Update last node position for next conversation turn
        self.last_node_position = {"x": center_x, "y": center_y}

    def draw_flow_layout(self, steps, conversation_turn):
        """Horizontal flowchart layout with conversation history"""
        start_x = 100 + (conversation_turn * 200)  # Shift right for each turn
        start_y = 300
        spacing = max(200, self.node_size * 2)
        
        for i, step in enumerate(steps):
            x = start_x
            y = start_y + (i * spacing)
            node = self.create_node(x, y, step[:20], "#FF6B6B" if i > 0 else "#2FA4FF", step)
            
            if i > 0:  # Draw connection line from previous node
                self.canvas.create_line(x, y - spacing, x, y, fill="#555555", width=2)
        
        # Update last node position
        self.last_node_position = {"x": start_x, "y": start_y + ((len(steps)-1) * spacing)}

    def draw_tree_layout(self, steps, conversation_turn):
        """Vertical tree layout with conversation history"""
        start_x = 100 + (conversation_turn * 200)  # Shift right for each turn
        start_y = 100
        level_spacing = max(150, self.node_size * 2)
        
        # Create root node
        main_node = steps[0] if steps else "Query"
        self.create_node(start_x, start_y, main_node[:20], "#2FA4FF", main_node)
        
        # Position child nodes
        num_children = len(steps) - 1
        if num_children > 0:
            child_spacing = 200
            for i, step in enumerate(steps[1:], 1):
                x = start_x - ((num_children-1) * child_spacing/2) + ((i-1) * child_spacing)
                y = start_y + level_spacing
                node = self.create_node(x, y, step[:20], "#FF6B6B", step)
                
                # Draw connection line
                self.canvas.create_line(start_x, start_y, x, y, fill="#555555", width=2)
        
        # Update last node position
        self.last_node_position = {"x": start_x, "y": start_y}

    def create_node(self, x, y, text, color, full_text=None):
        """Create a node with wrapped text and click handler"""
        node_id = self.canvas.create_oval(
            x-self.node_size, y-self.node_size,
            x+self.node_size, y+self.node_size,
            fill=color, outline="#404040", width=2
        )
        
        text_id = self.canvas.create_text(
            x, y,
            text=text,
            fill="white",
            width=self.node_size*1.8,
            justify=tk.CENTER,
            font=("Helvetica", 10)
        )
        
        if full_text:
            self.canvas.tag_bind(node_id, "<Button-1>", 
                lambda e, t=full_text: self.show_node_detail(t))
            self.canvas.tag_bind(text_id, "<Button-1>", 
                lambda e, t=full_text: self.show_node_detail(t))

    def show_node_detail(self, content):
        """Show detailed reasoning in response panel"""
        self.response_panel.configure(state="normal")
        self.response_panel.delete("1.0", "end")
        self.response_panel.insert("end", content)
        self.response_panel.configure(state="disabled")

    def process_input(self, event=None):
        """Handle query processing with visual updates"""
        if not self.input_entry.get().strip():
            return
            
        query = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        if not self.running:
            self.running = True
            self.canvas.delete("loading")  # Clear any existing loading animation
            self.show_processing_animation()
            
            # Start analysis in separate thread
            threading.Thread(target=self.run_analysis, args=(query,), daemon=True).start()

    def run_analysis(self, query):
        """Threaded analysis process"""
        try:
            # Get reasoning from model
            reasoning = self.model_chain.get_deepseek_reasoning(query)
            
            # Parse reasoning into steps
            steps = self.parse_reasoning_steps(reasoning)
            
            # Add reasoning to conversation history
            self.conversation_history.append({"role": "assistant", "content": reasoning})
            
            # Update canvas with new visualization
            self.canvas.after(0, lambda: self.canvas.delete("all"))  # Clear canvas
            
            # Calculate new node positions based on conversation history
            if self.current_view == "Radial":
                self.draw_radial_layout(steps, len(self.conversation_history))
            elif self.current_view == "Flow":
                self.draw_flow_layout(steps, len(self.conversation_history))
            else:  # Tree view
                self.draw_tree_layout(steps, len(self.conversation_history))
            
            # Get final response
            response = self.model_chain.get_groq_response(reasoning)
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Show response
            self.show_final_response(response)
            
        except Exception as e:
            self.show_error(str(e))
        finally:
            self.running = False
            self.canvas.delete("loading")

    def show_processing_animation(self):
        """Animated loading effect on canvas"""
        self.loading_active = True
        self.animation_angle = 0
        self.animate_loading()

    def animate_loading(self):
        if not self.loading_active:
            return
            
        center_x = self.canvas.winfo_width() // 2
        center_y = self.canvas.winfo_height() // 2
        radius = 30
        
        self.canvas.delete("loading")
        x = center_x + radius * math.cos(math.radians(self.animation_angle))
        y = center_y + radius * math.sin(math.radians(self.animation_angle))
        
        self.canvas.create_oval(
            x-5, y-5, x+5, y+5,
            fill="#2d2d2d", outline="#404040",
            tags="loading"
        )
        
        self.animation_angle = (self.animation_angle + 30) % 360
        self.after(50, self.animate_loading)

    def parse_reasoning_steps(self, reasoning_text):
        """Parse model output into structured steps"""
        steps = []
        current_step = []
        
        # Split on numbered items
        lines = reasoning_text.split("\n")
        for line in lines:
            if re.match(r"^\d+[\.\)]", line.strip()):
                if current_step:
                    steps.append(" ".join(current_step).strip())
                    current_step = []
                current_step.append(re.sub(r"^\d+[\.\)]\s*", "", line))
            else:
                current_step.append(line)
        
        if current_step:
            steps.append(" ".join(current_step).strip())
        
        return steps or [reasoning_text]

    def show_final_response(self, response):
        """Display final response in panel"""
        self.response_panel.configure(state="normal")
        self.response_panel.delete("1.0", "end")
        self.response_panel.insert("end", response)
        self.response_panel.configure(state="disabled")

    def change_view_mode(self, mode):
        """Handle view mode change"""
        self.current_view = mode
        if hasattr(self, "thinking_graph"):
            self.visualize_thought_process(self.thinking_graph.get("steps", []))

    def update_node_size(self, value):
        """Update node size from slider"""
        self.node_size = int(float(value))
        if hasattr(self, "thinking_graph"):
            self.visualize_thought_process(self.thinking_graph.get("steps", []))

    def change_reasoning_model(self, model_name):
        """Update reasoning model"""
        if self.model_chain.set_reasoning_model(model_name):
            self.show_node_detail(f"Reasoning model changed to {model_name}")

    def change_base_model(self, model_name):
        """Update base model"""
        if self.model_chain.set_base_model(model_name):
            self.show_node_detail(f"Base model changed to {model_name}")

    def show_error(self, message):
        """Display error message"""
        self.response_panel.configure(state="normal")
        self.response_panel.delete("1.0", "end")
        self.response_panel.insert("end", f"ERROR: {message}")
        self.response_panel.configure(state="disabled")

if __name__ == "__main__":
    load_dotenv()
    app = RatGUI()
    app.mainloop()