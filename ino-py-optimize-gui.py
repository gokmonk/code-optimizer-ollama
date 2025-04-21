import sys
import re
import os
import requests
import subprocess
import json
from datetime import datetime
import openai
from anthropic import Anthropic
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                            QTextEdit, QPushButton, QFileDialog, QLabel,
                            QHBoxLayout, QComboBox, QGroupBox, QRadioButton,
                            QMessageBox, QSplitter, QProgressBar, QTabWidget,
                            QToolBar, QStatusBar, QScrollArea, QToolButton, QDialog,
                            QLineEdit, QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QAction, QIcon, QTextOption, QColor, QPalette, QSyntaxHighlighter, QTextCharFormat
import time

class AIWorker(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(str, int)  # Message and percentage

    def __init__(self, prompt, ai_type, model, api_key=None):
        super().__init__()
        self.prompt = prompt
        self.ai_type = ai_type
        self.model = model
        self.api_key = api_key
        self.max_retries = 3
        self.timeout = 300  # 5 minutes timeout
        self.chunk_timeout = 30  # 30 seconds timeout between chunks
        self.is_running = True

    def run(self):
        try:
            if self.ai_type == 'ollama':
                self.handle_ollama_request()
            elif self.ai_type == 'deepseek':
                if not self.api_key:
                    raise Exception("DeepSeek API key is required")
                
                self.progress_updated.emit("Loading DeepSeek model...", 10)
                
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                    
                    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        "deepseek-ai/deepseek-coder-6.7b-instruct",
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16
                    ).cuda()
                    
                    self.progress_updated.emit("Processing with DeepSeek...", 30)
                    
                    messages = [{"role": "user", "content": self.prompt}]
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    self.progress_updated.emit("Generating response...", 60)
                    
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=4000,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=50,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    self.progress_updated.emit("Processing response...", 80)
                    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                    
                    if not result.strip():
                        raise Exception("Empty response from DeepSeek")
                    
                    self.progress_updated.emit("Finalizing results...", 90)
                    self.response_received.emit(result)
                    
                except Exception as e:
                    raise Exception(f"DeepSeek error: {str(e)}")
            
            elif self.ai_type == 'claude':
                if not self.api_key:
                    raise Exception("Claude API key is required")
                
                self.progress_updated.emit("Connecting to Claude...", 10)
                client = Anthropic(api_key=self.api_key)
                
                self.progress_updated.emit("Sending request to Claude...", 30)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": self.prompt}]
                )
                
                self.progress_updated.emit("Processing response...", 60)
                result = response.content[0].text
                
                if not result.strip():
                    raise Exception("Empty response from Claude")
                
                self.progress_updated.emit("Finalizing results...", 90)
                self.response_received.emit(result)
            
            elif self.ai_type == 'openai':
                if not self.api_key:
                    raise Exception("OpenAI API key is required")
                
                self.progress_updated.emit("Connecting to OpenAI...", 10)
                client = openai.OpenAI(api_key=self.api_key)
                
                self.progress_updated.emit("Sending request to OpenAI...", 30)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.prompt}],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                self.progress_updated.emit("Processing response...", 60)
                result = response.choices[0].message.content
                
                if not result.strip():
                    raise Exception("Empty response from OpenAI")
                
                self.progress_updated.emit("Finalizing results...", 90)
                self.response_received.emit(result)
            
            elif self.ai_type == 'chatgpt':
                if not self.api_key:
                    raise Exception("ChatGPT API key is required")
                
                self.progress_updated.emit("Connecting to ChatGPT...", 10)
                client = openai.OpenAI(api_key=self.api_key)
                
                self.progress_updated.emit("Sending request to ChatGPT...", 30)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.prompt}],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                self.progress_updated.emit("Processing response...", 60)
                result = response.choices[0].message.content
                
                if not result.strip():
                    raise Exception("Empty response from ChatGPT")
                    
                self.progress_updated.emit("Finalizing results...", 90)
                self.response_received.emit(result)
                
        except Exception as e:
            self.error_occurred.emit(f"{self.ai_type} error: {str(e)}")
    
    def handle_ollama_request(self):
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries and self.is_running:
            try:
                self.progress_updated.emit("Connecting to Ollama...", 10)
                
                # Check if Ollama is running
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code != 200:
                        raise Exception("Ollama service not responding properly")
                except requests.exceptions.RequestException:
                    raise Exception("Ollama service not running. Please start Ollama first.")
                
                # Check if model is available
                models = [model['name'] for model in response.json()['models']]
                if self.model not in models:
                    raise Exception(f"Model '{self.model}' not found. Available models: {', '.join(models)}")
                
                self.progress_updated.emit("Sending request to Ollama...", 20)
                
                # Prepare the request with timeout
                full_response = ""
                start_time = time.time()
                last_chunk_time = time.time()
                
                with requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": self.prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 50
                        }
                    },
                    stream=True,
                    timeout=self.timeout
                ) as response:
                    if response.status_code != 200:
                        raise Exception(f"API Error: {response.text}")
                    
                    self.progress_updated.emit("Processing response...", 30)
                    progress_step = 60 / 100  # Distribute 60% of progress bar during processing
                    received_chunks = 0
                    
                    for line in response.iter_lines():
                        if not self.is_running:
                            raise Exception("Operation cancelled by user")
                            
                        current_time = time.time()
                        if current_time - last_chunk_time > self.chunk_timeout:
                            raise Exception("Timeout waiting for response chunk")
                        last_chunk_time = current_time
                        
                        if current_time - start_time > self.timeout:
                            raise Exception("Overall request timeout exceeded")
                        
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    full_response += chunk['response']
                                    received_chunks += 1
                                    if received_chunks % 5 == 0:  # Update every 5 chunks
                                        progress = min(80, 30 + int(received_chunks * progress_step))
                                        self.progress_updated.emit(
                                            f"Processing... ({len(full_response)} chars, {received_chunks} chunks)",
                                            progress
                                        )
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON chunks
                
                if not full_response.strip():
                    raise Exception("Empty response from Ollama")
                
                self.progress_updated.emit("Finalizing results...", 90)
                self.response_received.emit(full_response)
                return  # Success, exit the retry loop
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < self.max_retries and self.is_running:
                    wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30 seconds
                    self.progress_updated.emit(
                        f"Retry {retry_count}/{self.max_retries} in {wait_time} seconds...",
                        10
                    )
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed after {retry_count} attempts. Last error: {last_error}")
    
    def stop(self):
        """Stop the worker thread"""
        self.is_running = False

class CodeEditor(QTextEdit):
    """Enhanced text editor with code-focused features"""
    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFont("Consolas, 'Source Code Pro', 'Courier New', monospace", 11)
        self.setFont(font)
        self.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' '))

class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        # Section headers
        header_format = QTextCharFormat()
        header_format.setForeground(QColor("#009900"))
        header_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((r'^(EXPLANATIONS|IMPROVED CODE):', header_format))
        
        # Separators
        separator_format = QTextCharFormat()
        separator_format.setForeground(QColor("#666666"))
        self.highlighting_rules.append((r'^-{80}$', separator_format))
        
        # Numbered explanations
        explanation_format = QTextCharFormat()
        explanation_format.setForeground(QColor("#0066CC"))
        self.highlighting_rules.append((r'^\d+\.\s+.*$', explanation_format))
        
        # Code blocks
        code_format = QTextCharFormat()
        code_format.setFontFamily("Consolas, 'Source Code Pro', 'Courier New', monospace")
        code_format.setBackground(QColor("#f8f8f8"))
        self.highlighting_rules.append((r'```(?:python|arduino)?\n(.*?)\n```', code_format))
        
        # Python syntax
        python_keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
            'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass',
            'raise', 'return', 'try', 'while', 'with', 'yield'
        ]
        python_format = QTextCharFormat()
        python_format.setForeground(QColor("#0000FF"))
        python_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((r'\b(' + '|'.join(python_keywords) + r')\b', python_format))
        
        # Python strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
        self.highlighting_rules.append((r"'[^'\\]*(\\.[^'\\]*)*'", string_format))
        
        # Python comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append((r'#.*$', comment_format))
        
        # Arduino syntax
        arduino_keywords = [
            'void', 'setup', 'loop', 'if', 'else', 'for', 'switch', 'case',
            'while', 'do', 'break', 'continue', 'return', 'goto', 'true', 'false',
            'HIGH', 'LOW', 'INPUT', 'OUTPUT', 'INPUT_PULLUP', 'LED_BUILTIN',
            'int', 'float', 'double', 'boolean', 'char', 'byte', 'word', 'long',
            'unsigned', 'short', 'String', 'array', 'sizeof', 'static', 'volatile',
            'const', 'struct', 'typedef', 'enum', 'extern', 'register', 'signed'
        ]
        arduino_format = QTextCharFormat()
        arduino_format.setForeground(QColor("#0000FF"))
        arduino_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((r'\b(' + '|'.join(arduino_keywords) + r')\b', arduino_format))
        
        # Arduino numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF00FF"))
        self.highlighting_rules.append((r'\b\d+\b', number_format))
        
        # Arduino comments
        arduino_comment_format = QTextCharFormat()
        arduino_comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append((r'//.*$', arduino_comment_format))
        self.highlighting_rules.append((r'/\*.*?\*/', arduino_comment_format))
        
        # Function calls
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#000080"))
        self.highlighting_rules.append((r'\b\w+(?=\()', function_format))
        
        # Variables
        variable_format = QTextCharFormat()
        variable_format.setForeground(QColor("#000000"))
        self.highlighting_rules.append((r'\b\w+\b', variable_format))
    
    def highlightBlock(self, text):
        # First, check if we're in a code block
        is_code_block = False
        if text.startswith('```'):
            is_code_block = True
            # Determine language
            if 'python' in text.lower():
                self.setCurrentBlockState(1)  # Python code
            elif 'arduino' in text.lower():
                self.setCurrentBlockState(2)  # Arduino code
            else:
                self.setCurrentBlockState(0)  # Unknown/plain text
        
        # Apply highlighting rules
        for pattern, format in self.highlighting_rules:
            for match in re.finditer(pattern, text, re.MULTILINE):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)
        
        # Special handling for code blocks
        if is_code_block:
            # Highlight the entire code block
            code_format = QTextCharFormat()
            code_format.setBackground(QColor("#f8f8f8"))
            self.setFormat(0, len(text), code_format)
            
            # Apply language-specific highlighting
            if self.currentBlockState() == 1:  # Python
                self.highlightPythonCode(text)
            elif self.currentBlockState() == 2:  # Arduino
                self.highlightArduinoCode(text)
    
    def highlightPythonCode(self, text):
        # Python-specific highlighting
        pass  # Already handled by the rules
    
    def highlightArduinoCode(self, text):
        # Arduino-specific highlighting
        pass  # Already handled by the rules

class PromptSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget for different prompt types
        self.tab_widget = QTabWidget()
        
        # Analysis prompt tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.analysis_prompt = QTextEdit()
        self.analysis_prompt.setPlaceholderText("Enter the prompt template for code analysis...")
        analysis_layout.addWidget(QLabel("Analysis Prompt Template:"))
        analysis_layout.addWidget(self.analysis_prompt)
        
        # Add AI improvement button for analysis
        analysis_btn_layout = QHBoxLayout()
        self.improve_analysis_btn = QPushButton("Ask AI to Improve")
        self.improve_analysis_btn.clicked.connect(lambda: self.improve_prompt("analysis"))
        analysis_btn_layout.addWidget(self.improve_analysis_btn)
        analysis_btn_layout.addStretch()
        analysis_layout.addLayout(analysis_btn_layout)
        
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        # Optimization prompt tab
        optimization_tab = QWidget()
        optimization_layout = QVBoxLayout(optimization_tab)
        self.optimization_prompt = QTextEdit()
        self.optimization_prompt.setPlaceholderText("Enter the prompt template for code optimization...")
        optimization_layout.addWidget(QLabel("Optimization Prompt Template:"))
        optimization_layout.addWidget(self.optimization_prompt)
        
        # Add AI improvement button for optimization
        optimization_btn_layout = QHBoxLayout()
        self.improve_optimization_btn = QPushButton("Ask AI to Improve")
        self.improve_optimization_btn.clicked.connect(lambda: self.improve_prompt("optimization"))
        optimization_btn_layout.addWidget(self.improve_optimization_btn)
        optimization_btn_layout.addStretch()
        optimization_layout.addLayout(optimization_btn_layout)
        
        self.tab_widget.addTab(optimization_tab, "Optimization")
        
        # Improvement prompt tab
        improvement_tab = QWidget()
        improvement_layout = QVBoxLayout(improvement_tab)
        self.improvement_prompt = QTextEdit()
        self.improvement_prompt.setPlaceholderText("Enter the prompt template for code improvement...")
        improvement_layout.addWidget(QLabel("Improvement Prompt Template:"))
        improvement_layout.addWidget(self.improvement_prompt)
        
        # Inline improvement prompt tab
        inline_tab = QWidget()
        inline_layout = QVBoxLayout(inline_tab)
        self.inline_prompt = QTextEdit()
        self.inline_prompt.setPlaceholderText("Enter the prompt template for inline code improvements...")
        inline_layout.addWidget(QLabel("Inline Improvement Prompt Template:"))
        inline_layout.addWidget(self.inline_prompt)
        
        # Add AI improvement button for improvement
        improvement_btn_layout = QHBoxLayout()
        self.improve_improvement_btn = QPushButton("Ask AI to Improve")
        self.improve_improvement_btn.clicked.connect(lambda: self.improve_prompt("improvement"))
        improvement_btn_layout.addWidget(self.improve_improvement_btn)
        improvement_btn_layout.addStretch()
        improvement_layout.addLayout(improvement_btn_layout)
        
        self.tab_widget.addTab(improvement_tab, "Improvement")
        
        # Add AI improvement button for inline
        inline_btn_layout = QHBoxLayout()
        self.improve_inline_btn = QPushButton("Ask AI to Improve")
        self.improve_inline_btn.clicked.connect(lambda: self.improve_prompt("inline"))
        inline_btn_layout.addWidget(self.improve_inline_btn)
        inline_btn_layout.addStretch()
        inline_layout.addLayout(inline_btn_layout)
        
        self.tab_widget.addTab(inline_tab, "Inline")
        
        layout.addWidget(self.tab_widget)
        
        # Add buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Load saved prompts or defaults
        self.load_prompts()
        
        # Create progress dialog for AI improvements
        self.progress_dialog = QProgressDialog("Improving prompt...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.hide()
    
    def improve_prompt(self, prompt_type):
        current_prompt = getattr(self, f"{prompt_type}_prompt").toPlainText()
        
        # Create a prompt to improve the current prompt
        improvement_prompt = f"""Please improve this prompt template for code {prompt_type}. 
The prompt should be clear, specific, and structured to get the best results from an AI model.
Focus on making it more effective while maintaining its core purpose.

Current prompt:
{current_prompt}

Please provide an improved version that:
1. Is more specific and detailed
2. Has better structure and formatting
3. Includes clear instructions for the AI
4. Maintains all necessary variables: {{language}}, {{scope}}, {{style_tag}}, {{compatibility_tag}}, {{best_practices_tag}}, {{code}}
5. Produces more consistent and high-quality results

Return only the improved prompt template, without any additional explanation."""
        
        # Show progress dialog
        self.progress_dialog.setLabelText(f"Improving {prompt_type} prompt...")
        self.progress_dialog.show()
        
        # Create and start AI worker
        self.worker = AIWorker(
            improvement_prompt,
            self.parent().current_ai,
            self.parent().current_model,
            self.parent().claude_api_key if self.parent().current_ai == 'claude' else 
            self.parent().openai_api_key if self.parent().current_ai == 'openai' else None
        )
        self.worker.response_received.connect(
            lambda response: self.handle_improvement_response(response, prompt_type))
        self.worker.error_occurred.connect(self.handle_improvement_error)
        self.worker.progress_updated.connect(self.update_improvement_progress)
        self.worker.start()
    
    def handle_improvement_response(self, response, prompt_type):
        # Update the prompt with the improved version
        getattr(self, f"{prompt_type}_prompt").setPlainText(response.strip())
        self.progress_dialog.hide()
        QMessageBox.information(self, "Success", f"{prompt_type.capitalize()} prompt improved successfully!")
    
    def handle_improvement_error(self, error):
        self.progress_dialog.hide()
        QMessageBox.critical(self, "Error", f"Failed to improve prompt: {error}")
    
    def update_improvement_progress(self, message, progress):
        self.progress_dialog.setLabelText(message)
        self.progress_dialog.setValue(progress)
    
    def load_prompts(self):
        # Try to load from settings file
        try:
            with open("prompt_settings.json", "r") as f:
                prompts = json.load(f)
                self.analysis_prompt.setText(prompts.get("analysis", ""))
                self.optimization_prompt.setText(prompts.get("optimization", ""))
                self.improvement_prompt.setText(prompts.get("improvement", ""))
                self.inline_prompt.setText(prompts.get("inline", ""))
        except FileNotFoundError:
            self.reset_to_defaults()
    
    def reset_to_defaults(self):
        # Default analysis prompt
        analysis_prompt = """Analyze this {language} code ({scope}) and provide detailed feedback:
1. {style_tag} style violations (highlight with [STYLE] tag)
2. Performance bottlenecks (highlight with [PERFORMANCE] tag)
3. Potential bugs (highlight with [BUG] tag)
4. Memory usage issues (highlight with [MEMORY] tag)
5. {compatibility_tag} (highlight with [COMPATIBILITY] tag)

Provide specific recommendations with line numbers where applicable.

{language} Code:
{code}"""
        
        # Default optimization prompt
        optimization_prompt = """Optimize this {language} code ({scope}) with:
1. {style_tag} style compliance
2. Performance improvements
3. Memory efficiency
4. {best_practices_tag}
5. Error handling improvements

Provide the complete rewritten code with explanations of key changes.
Format the response with clear section headers.

{language} Code:
{code}"""
        
        # Default improvement prompt
        improvement_prompt = """Improve this {language} code ({scope}) by implementing the following suggestions:
1. Fix {style_tag} style violations
2. Address performance bottlenecks
3. Fix potential bugs
4. Optimize memory usage
5. Ensure {compatibility_tag}

For each improvement:
- Show the original code
- Explain the issue
- Show the improved code
- Explain why the improvement is better

{language} Code:
{code}

Provide the complete improved code with explanations of key changes.
Format the response with clear section headers."""

        # Default inline improvement prompt
        inline_prompt = """Improve this {language} code ({scope}) with inline comments only.
Focus on:
1. Code optimization and performance improvements
2. Best practices and coding standards
3. Memory efficiency and resource management
4. Error handling and edge cases
5. Code readability and maintainability

Requirements:
- Add clear, concise inline comments explaining key improvements
- Keep comments brief and focused on the improvement
- Use standard comment style for {language}
- Maintain the original code structure
- Do not add any explanations outside the code
- Do not include any section headers or additional text

Return ONLY the improved code with inline comments in a code block.

{language} Code:
{code}"""
        
        self.analysis_prompt.setText(analysis_prompt)
        self.optimization_prompt.setText(optimization_prompt)
        self.improvement_prompt.setText(improvement_prompt)
        self.inline_prompt.setText(inline_prompt)
    
    def save_prompts(self):
        prompts = {
            "analysis": self.analysis_prompt.toPlainText(),
            "optimization": self.optimization_prompt.toPlainText(),
            "improvement": self.improvement_prompt.toPlainText(),
            "inline": self.inline_prompt.toPlainText()
        }
        with open("prompt_settings.json", "w") as f:
            json.dump(prompts, f, indent=4)

class CodeOptimizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Language Code Optimizer with AI")
        self.setGeometry(100, 100, 1200, 800)
        
        self.ollama_models = self.get_installed_ollama_models()
        self.deepseek_models = ["deepseek-coder-33b-instruct", "deepseek-coder-6.7b-instruct"]
        self.claude_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"]
        self.openai_models = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        self.chatgpt_models = ["gpt-4", "gpt-3.5-turbo"]
        
        self.current_model = self.ollama_models[0] if self.ollama_models else ""
        self.current_ai = "ollama" if self.ollama_models else "deepseek"
        self.current_file_type = "python"
        self.current_file_path = ""
        self.source_code = ""
        self.analysis_results = ""
        
        # Get API keys from environment variables
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self.chatgpt_api_key = os.getenv('CHATGPT_API_KEY', '')
        
        self.init_ui()
        
        # Load prompts
        self.load_prompts()
        
    def get_installed_ollama_models(self):
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output to get model names and sizes
            models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header line
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:  # Model name and size should be present
                        model_name = parts[0]
                        size_str = parts[2]  # Size is in the third column
                        # Convert size to bytes for consistent comparison
                        size_bytes = self.convert_size_to_bytes(size_str)
                        models.append((model_name, size_bytes))
            return models
        except (subprocess.CalledProcessError, FileNotFoundError):
            QMessageBox.warning(
                self,
                "Ollama Not Found",
                "Ollama not installed or not in PATH. Only DeepSeek models will be available."
            )
            return []

    def convert_size_to_bytes(self, size_str):
        """Convert size string (e.g., '1.2GB', '500MB') to bytes"""
        try:
            size = float(size_str[:-2])  # Extract the number
            unit = size_str[-2:].upper()  # Extract the unit (GB, MB)
            if unit == 'GB':
                return int(size * 1024 * 1024 * 1024)
            elif unit == 'MB':
                return int(size * 1024 * 1024)
            elif unit == 'KB':
                return int(size * 1024)
            return int(size)
        except (ValueError, IndexError):
            return 0

    def sort_ollama_models(self, models, sort_by='name'):
        """Sort Ollama models by name or size"""
        if sort_by == 'name':
            # Sort alphabetically by name
            return [model[0] for model in sorted(models, key=lambda x: x[0].lower())]
        else:  # sort_by == 'size'
            # Sort by size (largest first)
            return [model[0] for model in sorted(models, key=lambda x: x[1], reverse=True)]

    def init_ui(self):
        # Create main toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setStyleSheet("""
            QToolBar {
                border: none;
                background: #2d2d2d;
                spacing: 2px;
                padding: 1px;
            }
            QToolButton {
                padding: 2px 4px;
                border: 1px solid transparent;
                border-radius: 2px;
                color: #ffffff;
            }
            QToolButton:hover {
                background: #3d3d3d;
                border: 1px solid #4d4d4d;
            }
        """)
        self.addToolBar(self.toolbar)
        
        # Actions for toolbar
        self.action_load = QAction("Load File", self)
        self.action_load.triggered.connect(self.load_file)
        self.toolbar.addAction(self.action_load)
        
        self.action_analyze = QAction("Analyze", self)
        self.action_analyze.triggered.connect(self.analyze_code)
        self.toolbar.addAction(self.action_analyze)
        
        self.action_optimize = QAction("Optimize", self)
        self.action_optimize.triggered.connect(self.optimize_code)
        self.toolbar.addAction(self.action_optimize)
        
        self.action_improve = QAction("Improve Code", self)
        self.action_improve.triggered.connect(self.improve_code)
        self.toolbar.addAction(self.action_improve)
        
        self.action_improve_inline = QAction("Improve Code (Inline Only)", self)
        self.action_improve_inline.triggered.connect(self.improve_code_inline)
        self.toolbar.addAction(self.action_improve_inline)
        
        self.action_save = QAction("Save Results", self)
        self.action_save.triggered.connect(self.save_results)
        self.action_save.setEnabled(False)
        self.toolbar.addAction(self.action_save)
        
        self.action_settings = QAction("Settings", self)
        self.action_settings.triggered.connect(self.show_settings)
        self.toolbar.addAction(self.action_settings)
        
        # Add prompt settings action to toolbar
        self.action_prompt_settings = QAction("Prompt Settings", self)
        self.action_prompt_settings.triggered.connect(self.show_prompt_settings)
        self.toolbar.addAction(self.action_prompt_settings)
        
        # Add stop button to toolbar
        self.action_stop = QAction("Stop", self)
        self.action_stop.triggered.connect(self.stop_processing)
        self.action_stop.setEnabled(False)
        self.toolbar.addAction(self.action_stop)
        
        # Create central widget
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Settings panel (collapsible in the future)
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # First row of settings: AI Engine and Model selection
        ai_model_layout = QHBoxLayout()
        
        # AI Engine selection
        ai_group = QGroupBox("AI Engine")
        ai_layout = QHBoxLayout()
        
        self.ollama_radio = QRadioButton("Ollama")
        self.ollama_radio.setChecked(bool(self.ollama_models))
        self.ollama_radio.toggled.connect(self.update_ai_selection)
        self.ollama_radio.setEnabled(bool(self.ollama_models))
        ai_layout.addWidget(self.ollama_radio)
        
        self.deepseek_radio = QRadioButton("DeepSeek Coder")
        self.deepseek_radio.toggled.connect(self.update_ai_selection)
        self.deepseek_radio.setChecked(not bool(self.ollama_models))
        ai_layout.addWidget(self.deepseek_radio)
        
        self.claude_radio = QRadioButton("Claude")
        self.claude_radio.toggled.connect(self.update_ai_selection)
        ai_layout.addWidget(self.claude_radio)
        
        self.openai_radio = QRadioButton("OpenAI")
        self.openai_radio.toggled.connect(self.update_ai_selection)
        ai_layout.addWidget(self.openai_radio)
        
        self.chatgpt_radio = QRadioButton("ChatGPT")
        self.chatgpt_radio.toggled.connect(self.update_ai_selection)
        ai_layout.addWidget(self.chatgpt_radio)
        
        ai_group.setLayout(ai_layout)
        ai_model_layout.addWidget(ai_group)
        
        # Model selection
        model_group = QGroupBox("AI Model")
        model_layout = QVBoxLayout()  # Changed to VBoxLayout to accommodate sort options
        
        # Add sort options for Ollama models
        sort_layout = QHBoxLayout()
        sort_label = QLabel("Sort by:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Size"])
        self.sort_combo.currentTextChanged.connect(self.update_model_list)
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combo)
        model_layout.addLayout(sort_layout)
        
        model_select_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        if self.ollama_models:
            self.model_combo.addItems([model[0] for model in self.ollama_models])
        else:
            self.model_combo.addItems(self.deepseek_models)
        self.model_combo.currentTextChanged.connect(self.update_model)
        model_select_layout.addWidget(self.model_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_models)
        model_select_layout.addWidget(self.refresh_btn)
        
        model_layout.addLayout(model_select_layout)
        model_group.setLayout(model_layout)
        ai_model_layout.addWidget(model_group)
        
        settings_layout.addLayout(ai_model_layout)
        
        # Second row: File type and Analysis options
        file_analysis_layout = QHBoxLayout()
        
        # File Type Selection
        filetype_group = QGroupBox("File Type")
        filetype_layout = QHBoxLayout()
        
        self.python_radio = QRadioButton("Python")
        self.python_radio.setChecked(True)
        self.python_radio.toggled.connect(self.update_file_type)
        filetype_layout.addWidget(self.python_radio)
        
        self.arduino_radio = QRadioButton("Arduino")
        self.arduino_radio.toggled.connect(self.update_file_type)
        filetype_layout.addWidget(self.arduino_radio)
        
        filetype_group.setLayout(filetype_layout)
        file_analysis_layout.addWidget(filetype_group)
        
        # Analysis options
        options_group = QGroupBox("Analysis Scope")
        options_layout = QHBoxLayout()
        
        self.full_file_check = QRadioButton("Entire File")
        self.full_file_check.setChecked(True)
        options_layout.addWidget(self.full_file_check)
        
        self.class_check = QRadioButton("Selected Class" if self.current_file_type != "arduino" else "Selected Setup/Loop")
        options_layout.addWidget(self.class_check)
        
        self.function_check = QRadioButton("Selected Function")
        options_layout.addWidget(self.function_check)
        
        self.custom_check = QRadioButton("Custom Selection")
        options_layout.addWidget(self.custom_check)
        
        options_group.setLayout(options_layout)
        file_analysis_layout.addWidget(options_group)
        
        settings_layout.addLayout(file_analysis_layout)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #e0e0e0;
                height: 2px;
            }
            QSplitter::handle:hover {
                background: #c0c0c0;
            }
        """)
        
        # Output tabs (Analysis, Optimized Code)
        self.output_tabs = QTabWidget()
        self.output_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.output_tabs.setDocumentMode(True)
        self.output_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                top: -1px;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-bottom: none;
                padding: 2px 6px;
                margin-right: 1px;
                font-size: 11px;
                min-width: 80px;
                color: #cccccc;
            }
            QTabBar::tab:selected {
                background: #1e1e1e;
                border-bottom-color: #1e1e1e;
            }
            QTabBar::tab:hover {
                background: #3d3d3d;
            }
        """)
        
        # Source code tab
        source_container = QWidget()
        source_layout = QVBoxLayout(source_container)
        source_layout.setContentsMargins(1, 1, 1, 1)
        source_layout.setSpacing(1)
        
        self.input_code = CodeEditor()
        self.input_code.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #1e1e1e;
                color: #d4d4d4;
                selection-background-color: #264f78;
            }
        """)
        self.input_code.setPlaceholderText("Paste your code here or load from file...")
        self.input_highlighter = CodeHighlighter(self.input_code.document())
        source_layout.addWidget(self.input_code)
        
        current_file_layout = QHBoxLayout()
        current_file_layout.setContentsMargins(1, 1, 1, 1)
        current_file_layout.setSpacing(2)
        current_file_label = QLabel("Current file:")
        current_file_label.setStyleSheet("font-size: 10px; color: #888888;")
        current_file_label.setMaximumHeight(16)
        self.current_file_label = QLabel("No file loaded")
        self.current_file_label.setStyleSheet("font-size: 10px; color: #888888;")
        self.current_file_label.setMaximumHeight(16)
        current_file_layout.addWidget(current_file_label)
        current_file_layout.addWidget(self.current_file_label)
        current_file_layout.addStretch()
        source_layout.addLayout(current_file_layout)
        
        self.output_tabs.addTab(source_container, "Source Code")
        
        # Analysis tab
        self.output_display = CodeEditor()
        self.output_display.setReadOnly(True)
        self.output_display.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #1e1e1e;
                color: #d4d4d4;
                selection-background-color: #264f78;
            }
        """)
        self.output_highlighter = CodeHighlighter(self.output_display.document())
        self.output_tabs.addTab(self.output_display, "Analysis/Optimization Results")
        
        # Optimized code tab
        self.optimized_code = CodeEditor()
        self.optimized_code.setReadOnly(True)
        self.optimized_code.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #1e1e1e;
                color: #d4d4d4;
                selection-background-color: #264f78;
            }
        """)
        self.optimized_highlighter = CodeHighlighter(self.optimized_code.document())
        self.output_tabs.addTab(self.optimized_code, "Optimized Code")
        
        # Improved code tab
        improved_container = QWidget()
        improved_layout = QVBoxLayout(improved_container)
        improved_layout.setContentsMargins(1, 1, 1, 1)
        improved_layout.setSpacing(1)
        
        improved_toolbar = QToolBar()
        improved_toolbar.setIconSize(QSize(16, 16))
        improved_toolbar.setMaximumHeight(20)
        improved_toolbar.setStyleSheet("""
            QToolBar {
                border: none;
                background: transparent;
                spacing: 2px;
            }
            QToolButton {
                padding: 1px 3px;
                border: 1px solid transparent;
                border-radius: 2px;
                font-size: 10px;
                color: #cccccc;
            }
            QToolButton:hover {
                background: #3d3d3d;
                border: 1px solid #4d4d4d;
            }
        """)
        
        self.copy_improved_btn = QToolButton()
        self.copy_improved_btn.setText("Copy Code")
        self.copy_improved_btn.clicked.connect(self.copy_improved_code)
        improved_toolbar.addWidget(self.copy_improved_btn)
        
        self.show_diff_btn = QToolButton()
        self.show_diff_btn.setText("Show Diff")
        self.show_diff_btn.clicked.connect(self.show_diff_view)
        improved_toolbar.addWidget(self.show_diff_btn)
        
        improved_layout.addWidget(improved_toolbar)
        
        self.improved_code = CodeEditor()
        self.improved_code.setReadOnly(True)
        self.improved_code.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #1e1e1e;
                color: #d4d4d4;
                selection-background-color: #264f78;
            }
        """)
        self.improved_highlighter = CodeHighlighter(self.improved_code.document())
        improved_layout.addWidget(self.improved_code)
        
        self.output_tabs.addTab(improved_container, "Improved Code")
        
        # Inline improved code tab
        inline_container = QWidget()
        inline_layout = QVBoxLayout(inline_container)
        inline_layout.setContentsMargins(1, 1, 1, 1)
        inline_layout.setSpacing(1)
        
        inline_toolbar = QToolBar()
        inline_toolbar.setIconSize(QSize(16, 16))
        inline_toolbar.setMaximumHeight(20)
        inline_toolbar.setStyleSheet(improved_toolbar.styleSheet())
        
        self.copy_inline_btn = QToolButton()
        self.copy_inline_btn.setText("Copy Code")
        self.copy_inline_btn.clicked.connect(self.copy_inline_code)
        inline_toolbar.addWidget(self.copy_inline_btn)
        
        self.show_inline_diff_btn = QToolButton()
        self.show_inline_diff_btn.setText("Show Diff")
        self.show_inline_diff_btn.clicked.connect(self.show_inline_diff_view)
        inline_toolbar.addWidget(self.show_inline_diff_btn)
        
        inline_layout.addWidget(inline_toolbar)
        
        self.inline_code = CodeEditor()
        self.inline_code.setReadOnly(True)
        self.inline_code.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #1e1e1e;
                color: #d4d4d4;
                selection-background-color: #264f78;
            }
        """)
        self.inline_highlighter = CodeHighlighter(self.inline_code.document())
        inline_layout.addWidget(self.inline_code)
        
        self.output_tabs.addTab(inline_container, "Inline Improved Code")
        
        main_layout.addWidget(self.output_tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)  # Hidden by default
        main_layout.addWidget(self.progress_bar)
        
        # Status bar at the bottom of the window
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, 1)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def update_file_type(self):
        if self.python_radio.isChecked():
            self.current_file_type = "python"
            self.class_check.setText("Selected Class")
            # Update highlighter rules for Python
            self.update_highlighter_rules(is_python=True)
        else:
            self.current_file_type = "arduino"
            self.class_check.setText("Selected Setup/Loop")
            # Update highlighter rules for Arduino
            self.update_highlighter_rules(is_python=False)
    
    def update_highlighter_rules(self, is_python=True):
        # Update all highlighters with new rules
        highlighters = [
            self.input_highlighter,
            self.output_highlighter,
            self.optimized_highlighter,
            self.improved_highlighter,
            self.inline_highlighter
        ]
        
        for highlighter in highlighters:
            highlighter.highlighting_rules.clear()
            
            # Common rules
            header_format = QTextCharFormat()
            header_format.setForeground(QColor("#009900"))
            header_format.setFontWeight(QFont.Weight.Bold)
            highlighter.highlighting_rules.append((r'^(EXPLANATIONS|IMPROVED CODE):', header_format))
            
            separator_format = QTextCharFormat()
            separator_format.setForeground(QColor("#666666"))
            highlighter.highlighting_rules.append((r'^-{80}$', separator_format))
            
            explanation_format = QTextCharFormat()
            explanation_format.setForeground(QColor("#0066CC"))
            highlighter.highlighting_rules.append((r'^\d+\.\s+.*$', explanation_format))
            
            # Code blocks
            code_format = QTextCharFormat()
            code_format.setFontFamily("Consolas, 'Source Code Pro', 'Courier New', monospace")
            code_format.setBackground(QColor("#2d2d2d"))
            highlighter.highlighting_rules.append((r'```(?:python|arduino)?\n(.*?)\n```', code_format))
            
            # Comments
            comment_format = QTextCharFormat()
            comment_format.setForeground(QColor("#6A9955"))
            
            # Strings
            string_format = QTextCharFormat()
            string_format.setForeground(QColor("#CE9178"))
            highlighter.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
            highlighter.highlighting_rules.append((r"'[^'\\]*(\\.[^'\\]*)*'", string_format))
            
            # Numbers
            number_format = QTextCharFormat()
            number_format.setForeground(QColor("#B5CEA8"))
            highlighter.highlighting_rules.append((r'\b\d+\b', number_format))
            
            if is_python:
                # Python keywords
                keywords = [
                    'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
                    'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
                    'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass',
                    'raise', 'return', 'try', 'while', 'with', 'yield'
                ]
                keyword_format = QTextCharFormat()
                keyword_format.setForeground(QColor("#569CD6"))
                keyword_format.setFontWeight(QFont.Weight.Bold)
                highlighter.highlighting_rules.append((r'\b(' + '|'.join(keywords) + r')\b', keyword_format))
                
                # Python comments
                highlighter.highlighting_rules.append((r'#.*$', comment_format))
                
            else:
                # Arduino keywords
                keywords = [
                    'void', 'setup', 'loop', 'if', 'else', 'for', 'switch', 'case',
                    'while', 'do', 'break', 'continue', 'return', 'goto', 'true', 'false',
                    'HIGH', 'LOW', 'INPUT', 'OUTPUT', 'INPUT_PULLUP', 'LED_BUILTIN',
                    'int', 'float', 'double', 'boolean', 'char', 'byte', 'word', 'long',
                    'unsigned', 'short', 'String', 'array', 'sizeof', 'static', 'volatile',
                    'const', 'struct', 'typedef', 'enum', 'extern', 'register', 'signed'
                ]
                keyword_format = QTextCharFormat()
                keyword_format.setForeground(QColor("#569CD6"))
                keyword_format.setFontWeight(QFont.Weight.Bold)
                highlighter.highlighting_rules.append((r'\b(' + '|'.join(keywords) + r')\b', keyword_format))
                
                # Arduino comments
                highlighter.highlighting_rules.append((r'//.*$', comment_format))
                highlighter.highlighting_rules.append((r'/\*.*?\*/', comment_format))
            
            # Function calls
            function_format = QTextCharFormat()
            function_format.setForeground(QColor("#DCDCAA"))
            highlighter.highlighting_rules.append((r'\b\w+(?=\()', function_format))
            
            # Variables
            variable_format = QTextCharFormat()
            variable_format.setForeground(QColor("#9CDCFE"))
            highlighter.highlighting_rules.append((r'\b\w+\b', variable_format))
            
            highlighter.rehighlight()
    
    def update_model_list(self):
        if self.ollama_radio.isChecked():
            sort_by = 'size' if self.sort_combo.currentText() == 'Size' else 'name'
            sorted_models = self.sort_ollama_models(self.ollama_models, sort_by)
            current_model = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(sorted_models)
            # Try to restore the previously selected model
            index = self.model_combo.findText(current_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

    def refresh_models(self):
        if self.ollama_radio.isChecked():
            self.ollama_models = self.get_installed_ollama_models()
            current_text = self.model_combo.currentText()
            self.update_model_list()
            # Try to restore the previously selected model
            index = self.model_combo.findText(current_text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            self.update_status("Ollama models refreshed", 0)
    
    def update_ai_selection(self):
        if self.ollama_radio.isChecked():
            self.current_ai = "ollama"
            self.model_combo.clear()
            self.sort_combo.setEnabled(True)
            self.update_model_list()
            self.refresh_btn.setEnabled(True)
        elif self.deepseek_radio.isChecked():
            self.current_ai = "deepseek"
            self.model_combo.clear()
            self.model_combo.addItems(self.deepseek_models)
            self.sort_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
        elif self.claude_radio.isChecked():
            self.current_ai = "claude"
            self.model_combo.clear()
            self.model_combo.addItems(self.claude_models)
            self.sort_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
        elif self.openai_radio.isChecked():
            self.current_ai = "openai"
            self.model_combo.clear()
            self.model_combo.addItems(self.openai_models)
            self.sort_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
        elif self.chatgpt_radio.isChecked():
            self.current_ai = "chatgpt"
            self.model_combo.clear()
            self.model_combo.addItems(self.chatgpt_models)
            self.sort_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
        self.current_model = self.model_combo.currentText()
    
    def update_model(self, model):
        self.current_model = model
    
    def load_file(self):
        file_types = "Python Files (*.py);;Arduino Files (*.ino);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", file_types
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.source_code = f.read()
                    self.input_code.setPlainText(self.source_code)
                self.current_file_path = file_path
                self.current_file_label.setText(file_path)
                self.update_status(f"Loaded: {os.path.basename(file_path)}", 0)
                
                # Auto-detect file type by extension
                if file_path.lower().endswith('.py'):
                    self.python_radio.setChecked(True)
                else:
                    self.arduino_radio.setChecked(True)
                self.update_file_type()
            except Exception as e:
                self.update_status(f"Error loading file: {str(e)}", 0)
                QMessageBox.critical(self, "File Load Error", f"Failed to load file: {str(e)}")
    
    def get_selected_code(self):
        full_code = self.input_code.toPlainText()
        
        if self.custom_check.isChecked():
            return self.input_code.textCursor().selectedText()
        
        if self.full_file_check.isChecked():
            return full_code
        
        cursor_pos = self.input_code.textCursor().position()
        
        if self.current_file_type == "python":
            if self.class_check.isChecked():
                pattern = r'(?:^|\n)\s*class\s+\w+.*?:\s*\n(?:[ \t]+.*\n)*'
            else:  # function
                pattern = r'(?:^|\n)\s*def\s+\w+\s*\(.*?\)\s*:\s*\n(?:[ \t]+.*\n)*'
        else:  # arduino
            if self.class_check.isChecked():  # For Arduino, this is setup/loop
                pattern = r'(?:^|\n)\s*(void\s+(setup|loop)\s*\(\)\s*\{.*?\n\})'
            else:  # function
                pattern = r'(?:^|\n)\s*(?:void|int|float|double|bool|char|byte|word|long|unsigned\s+)*\w+\s*\(.*?\)\s*\{.*?\n\}'
        
        matches = list(re.finditer(pattern, full_code, re.DOTALL))
        if not matches:
            return None
        
        containing_block = None
        for match in reversed(matches):
            if match.start() < cursor_pos:
                containing_block = match.group(0)
                break
        
        return containing_block
    
    def get_current_scope(self):
        if self.class_check.isChecked():
            if self.current_file_type == "python":
                return "selected class"
            else:
                return "selected setup/loop"
        elif self.function_check.isChecked():
            return "selected function"
        elif self.custom_check.isChecked():
            return "selected portion"
        return "entire file"
    
    def analyze_code(self):
        try:
            selected_code = self.get_selected_code()
            if not selected_code or not selected_code.strip():
                self.output_display.setPlainText("Please select valid code to analyze.")
                self.output_tabs.setCurrentIndex(0)
                return
            
            scope = self.get_current_scope()
            language = "Python" if self.current_file_type == "python" else "Arduino"
            style_tag = "PEP 8" if self.current_file_type == "python" else "Arduino"
            compatibility_tag = "Python version compatibility" if self.current_file_type == "python" else "Hardware compatibility"
            
            prompt = self.prompts["analysis"].format(
                language=language,
                scope=scope,
                style_tag=style_tag,
                compatibility_tag=compatibility_tag,
                code=selected_code[:10000]
            )
            
            self.output_display.setPlainText(f"Analyzing {scope}...")
            self.output_tabs.setCurrentIndex(0)
            self.update_status(f"Analyzing {scope} with {self.current_ai} ({self.current_model})...", 0)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.run_ai_query(prompt, is_optimization=False)
            
        except Exception as e:
            self.output_display.setPlainText(f"Error: {str(e)}")
            self.update_status("Analysis failed", 0)
            self.progress_bar.setVisible(False)
    
    def optimize_code(self):
        try:
            selected_code = self.get_selected_code()
            if not selected_code or not selected_code.strip():
                self.output_display.setPlainText("Please select valid code to optimize.")
                self.output_tabs.setCurrentIndex(0)
                return
                
            scope = self.get_current_scope()
            language = "Python" if self.current_file_type == "python" else "Arduino"
            style_tag = "PEP 8" if self.current_file_type == "python" else "Arduino"
            best_practices_tag = "Pythonic best practices" if self.current_file_type == "python" else "Arduino best practices"
            
            prompt = self.prompts["optimization"].format(
                language=language,
                scope=scope,
                style_tag=style_tag,
                best_practices_tag=best_practices_tag,
                code=selected_code[:10000]
            )
            
            self.output_display.setPlainText(f"Optimizing {scope}...")
            self.output_tabs.setCurrentIndex(0)
            self.update_status(f"Optimizing {scope} with {self.current_ai} ({self.current_model})...", 0)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.run_ai_query(prompt, is_optimization=True)
            
        except Exception as e:
            self.output_display.setPlainText(f"Error: {str(e)}")
            self.update_status("Optimization failed", 0)
            self.progress_bar.setVisible(False)
    
    def improve_code(self):
        try:
            selected_code = self.get_selected_code()
            if not selected_code or not selected_code.strip():
                self.output_display.setPlainText("Please select valid code to improve.")
                self.output_tabs.setCurrentIndex(0)
                return
            
            scope = self.get_current_scope()
            language = "Python" if self.current_file_type == "python" else "Arduino"
            style_tag = "PEP 8" if self.current_file_type == "python" else "Arduino"
            compatibility_tag = "Python version compatibility" if self.current_file_type == "python" else "Hardware compatibility"
            best_practices_tag = "Pythonic best practices" if self.current_file_type == "python" else "Arduino best practices"
            
            prompt = self.prompts["improvement"].format(
                language=language,
                scope=scope,
                style_tag=style_tag,
                compatibility_tag=compatibility_tag,
                best_practices_tag=best_practices_tag,
                code=selected_code[:10000]
            )
            
            self.output_display.setPlainText(f"Improving {scope}...")
            self.output_tabs.setCurrentIndex(0)
            self.update_status(f"Improving {scope} with {self.current_ai} ({self.current_model})...", 0)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.run_ai_query(prompt, is_optimization=True)
            
        except Exception as e:
            self.output_display.setPlainText(f"Error: {str(e)}")
            self.update_status("Improvement failed", 0)
            self.progress_bar.setVisible(False)
    
    def improve_code_inline(self):
        try:
            selected_code = self.get_selected_code()
            if not selected_code or not selected_code.strip():
                self.output_display.setPlainText("Please select valid code to improve.")
                self.output_tabs.setCurrentIndex(0)
                return
            
            scope = self.get_current_scope()
            language = "Python" if self.current_file_type == "python" else "Arduino"
            style_tag = "PEP 8" if self.current_file_type == "python" else "Arduino"
            
            prompt = self.prompts["inline"].format(
                language=language,
                scope=scope,
                style_tag=style_tag,
                code=selected_code[:10000]
            )
            
            self.output_display.setPlainText(f"Improving {scope} with inline comments...")
            self.output_tabs.setCurrentIndex(0)
            self.update_status(f"Improving {scope} with {self.current_ai} ({self.current_model})...", 0)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.run_ai_query(prompt, is_optimization=True)
            
        except Exception as e:
            self.output_display.setPlainText(f"Error: {str(e)}")
            self.update_status("Improvement failed", 0)
            self.progress_bar.setVisible(False)
    
    def run_ai_query(self, prompt, is_optimization=False):
        api_key = None
        if self.current_ai == 'claude':
            api_key = self.claude_api_key
        elif self.current_ai == 'openai':
            api_key = self.openai_api_key
        elif self.current_ai == 'chatgpt':
            api_key = self.chatgpt_api_key
            
        self.worker = AIWorker(prompt, self.current_ai, self.current_model, api_key)
        self.worker.response_received.connect(lambda response: self.handle_response(response, is_optimization))
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.progress_updated.connect(self.update_progress)
        self.action_stop.setEnabled(True)
        self.set_buttons_enabled(False)
        self.worker.start()
    
    def stop_processing(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.action_stop.setEnabled(False)
            self.set_buttons_enabled(True)
            self.progress_bar.setVisible(False)
            self.update_status("Operation cancelled by user", 0)
    
    def handle_error(self, error):
        self.output_display.setPlainText(f"Error: {error}")
        self.update_status("Error occurred during processing", 0)
        self.set_buttons_enabled(True)
        self.action_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def handle_response(self, response, is_optimization):
        try:
            self.analysis_results = response
            
            # If this was an inline improvement request
            if hasattr(self, 'inline_code') and "inline" in response.lower():
                # Extract code block from response
                code_blocks = re.findall(r'```(?:python|arduino)?\n(.*?)\n```', response, re.DOTALL)
                if code_blocks:
                    self.inline_code.setPlainText(code_blocks[0])
                else:
                    # If no code block found, use the entire response
                    self.inline_code.setPlainText(response)
                self.output_tabs.setTabText(3, "Inline Improved Code ✓")
                self.output_tabs.setCurrentIndex(3)
            else:
                self.output_display.setPlainText(response)
                
                # If this was an optimization request, try to extract the code
                if is_optimization:
                    optimized_code = self.extract_optimized_code(response)
                    self.optimized_code.setPlainText(optimized_code)
                    self.output_tabs.setTabText(1, "Optimized Code ✓")
                    
                    # If this was an improvement request, also update the improved code tab
                    if hasattr(self, 'improved_code'):
                        improved_code = self.extract_improved_code(response)
                        self.improved_code.setPlainText(improved_code)
                        self.output_tabs.setTabText(2, "Improved Code ✓")
                else:
                    self.output_tabs.setTabText(1, "Optimized Code")
                    if hasattr(self, 'improved_code'):
                        self.output_tabs.setTabText(2, "Improved Code")
            
            self.update_status("Analysis complete!", 100)
            self.action_save.setEnabled(True)
            self.set_buttons_enabled(True)
            self.action_stop.setEnabled(False)
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.output_display.setPlainText(f"Error processing response: {str(e)}")
            self.update_status("Error processing response", 0)
            self.set_buttons_enabled(True)
            self.action_stop.setEnabled(False)
            self.progress_bar.setVisible(False)
    
    def extract_optimized_code(self, analysis_results):
        # Try to extract code blocks from the AI response
        code_blocks = re.findall(r'```(?:python|arduino)?\n(.*?)\n```', analysis_results, re.DOTALL)
        if code_blocks:
            return code_blocks[-1]  # Return the last code block (usually the optimized version)
        return analysis_results  # Fallback to returning the full response
    
    def extract_improved_code(self, analysis_results):
        # Try to extract code blocks from the AI response
        code_blocks = re.findall(r'```(?:python|arduino)?\n(.*?)\n```', analysis_results, re.DOTALL)
        if code_blocks:
            # Extract explanations and improvements
            explanations = []
            improvements = []
            
            # Look for explanation sections
            explanation_sections = re.finditer(r'(?:Explanation|Improvement|Change):(.*?)(?=\n\n|$)', analysis_results, re.DOTALL)
            for section in explanation_sections:
                explanations.append(section.group(1).strip())
            
            # Look for improvement sections
            improvement_sections = re.finditer(r'(?:Improved|Updated|New) Code:(.*?)(?=\n\n|$)', analysis_results, re.DOTALL)
            for section in improvement_sections:
                improvements.append(section.group(1).strip())
            
            # Format the output
            output = []
            if explanations:
                output.append("EXPLANATIONS:")
                output.append("-" * 80)
                for i, exp in enumerate(explanations, 1):
                    output.append(f"{i}. {exp}\n")
            
            if improvements:
                output.append("\nIMPROVED CODE:")
                output.append("-" * 80)
                for imp in improvements:
                    output.append(imp)
            
            if not output:  # If no structured sections found, use the code blocks
                output.append("IMPROVED CODE:")
                output.append("-" * 80)
                output.append(code_blocks[-1])
            
            return "\n".join(output)
        return analysis_results  # Fallback to returning the full response
    
    def update_status(self, message, progress=None):
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)
    
    def update_progress(self, message, progress):
        """Update the progress bar and status label with the current progress"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def set_buttons_enabled(self, enabled):
        self.action_load.setEnabled(enabled)
        self.action_analyze.setEnabled(enabled)
        self.action_optimize.setEnabled(enabled)
        self.action_improve.setEnabled(enabled)
        self.action_improve_inline.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled and self.ollama_radio.isChecked())

    def copy_improved_code(self):
        clipboard = QApplication.clipboard()
        improved_text = self.improved_code.toPlainText()
        if improved_text.strip():
            clipboard.setText(improved_text)
            self.update_status("Improved code copied to clipboard", 0)
        else:
            QMessageBox.warning(self, "Warning", "No improved code to copy")

    def show_diff_view(self):
        if not self.source_code or not self.improved_code.toPlainText().strip():
            QMessageBox.warning(self, "Warning", "No code to compare")
            return
        
        # Extract just the code part from improved code
        improved_text = self.improved_code.toPlainText()
        code_match = re.search(r'IMPROVED CODE:.*?\n-{80}\n(.*?)$', improved_text, re.DOTALL)
        if not code_match:
            QMessageBox.warning(self, "Warning", "Could not extract improved code for comparison")
            return
        
        improved_code = code_match.group(1).strip()
        
        # Create a new dialog for diff view
        diff_dialog = QDialog(self)
        diff_dialog.setWindowTitle("Code Comparison")
        diff_dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(diff_dialog)
        
        # Create diff display
        diff_display = QTextEdit()
        diff_display.setReadOnly(True)
        diff_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas, 'Source Code Pro', 'Courier New', monospace';
                font-size: 11px;
            }
        """)
        
        # Simple diff implementation
        original_lines = self.source_code.split('\n')
        improved_lines = improved_code.split('\n')
        
        diff_text = []
        for i, (orig, imp) in enumerate(zip(original_lines, improved_lines)):
            if orig != imp:
                diff_text.append(f"Line {i+1}:")
                diff_text.append(f"- {orig}")
                diff_text.append(f"+ {imp}")
                diff_text.append("")
        
        diff_display.setPlainText('\n'.join(diff_text))
        layout.addWidget(diff_display)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(diff_dialog.close)
        layout.addWidget(close_btn)
        
        diff_dialog.exec()

    def copy_inline_code(self):
        clipboard = QApplication.clipboard()
        inline_text = self.inline_code.toPlainText()
        if inline_text.strip():
            clipboard.setText(inline_text)
            self.update_status("Inline improved code copied to clipboard", 0)
        else:
            QMessageBox.warning(self, "Warning", "No inline improved code to copy")

    def show_inline_diff_view(self):
        if not self.source_code or not self.inline_code.toPlainText().strip():
            QMessageBox.warning(self, "Warning", "No code to compare")
            return
        
        # Create a new dialog for inline diff view
        diff_dialog = QDialog(self)
        diff_dialog.setWindowTitle("Inline Code Comparison")
        diff_dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(diff_dialog)
        
        # Create diff display
        diff_display = QTextEdit()
        diff_display.setReadOnly(True)
        diff_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas, 'Source Code Pro', 'Courier New', monospace';
                font-size: 11px;
            }
        """)
        
        # Simple diff implementation
        original_lines = self.source_code.split('\n')
        inline_lines = self.inline_code.toPlainText().split('\n')
        
        diff_text = []
        for i, (orig, inline) in enumerate(zip(original_lines, inline_lines)):
            if orig != inline:
                diff_text.append(f"Line {i+1}:")
                diff_text.append(f"- {orig}")
                diff_text.append(f"+ {inline}")
                diff_text.append("")
        
        diff_display.setPlainText('\n'.join(diff_text))
        layout.addWidget(diff_display)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(diff_dialog.close)
        layout.addWidget(close_btn)
        
        diff_dialog.exec()

    def show_settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("API Settings")
        settings_dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(settings_dialog)
        
        # Claude API Key
        claude_group = QGroupBox("Claude API Settings")
        claude_layout = QVBoxLayout()
        
        claude_key_label = QLabel("API Key (from ANTHROPIC_API_KEY environment variable):")
        claude_key_input = QLineEdit()
        claude_key_input.setText(self.claude_api_key)
        claude_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        claude_key_input.setReadOnly(True)
        claude_key_input.setPlaceholderText("Set ANTHROPIC_API_KEY environment variable")
        
        claude_layout.addWidget(claude_key_label)
        claude_layout.addWidget(claude_key_input)
        claude_group.setLayout(claude_layout)
        layout.addWidget(claude_group)
        
        # OpenAI API Key
        openai_group = QGroupBox("OpenAI API Settings")
        openai_layout = QVBoxLayout()
        
        openai_key_label = QLabel("API Key (from OPENAI_API_KEY environment variable):")
        openai_key_input = QLineEdit()
        openai_key_input.setText(self.openai_api_key)
        openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        openai_key_input.setReadOnly(True)
        openai_key_input.setPlaceholderText("Set OPENAI_API_KEY environment variable")
        
        openai_layout.addWidget(openai_key_label)
        openai_layout.addWidget(openai_key_input)
        openai_group.setLayout(openai_layout)
        layout.addWidget(openai_group)
        
        # ChatGPT API Key
        chatgpt_group = QGroupBox("ChatGPT API Settings")
        chatgpt_layout = QVBoxLayout()
        
        chatgpt_key_label = QLabel("API Key (from CHATGPT_API_KEY environment variable):")
        chatgpt_key_input = QLineEdit()
        chatgpt_key_input.setText(self.chatgpt_api_key)
        chatgpt_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        chatgpt_key_input.setReadOnly(True)
        chatgpt_key_input.setPlaceholderText("Set CHATGPT_API_KEY environment variable")
        
        chatgpt_layout.addWidget(chatgpt_key_label)
        chatgpt_layout.addWidget(chatgpt_key_input)
        chatgpt_group.setLayout(chatgpt_layout)
        layout.addWidget(chatgpt_group)
        
        # DeepSeek API Key
        deepseek_group = QGroupBox("DeepSeek API Settings")
        deepseek_layout = QVBoxLayout()
        
        deepseek_key_label = QLabel("API Key (from DEEPSEEK_API_KEY environment variable):")
        deepseek_key_input = QLineEdit()
        deepseek_key_input.setText(self.deepseek_api_key)
        deepseek_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        deepseek_key_input.setReadOnly(True)
        deepseek_key_input.setPlaceholderText("Set DEEPSEEK_API_KEY environment variable")
        
        deepseek_layout.addWidget(deepseek_key_label)
        deepseek_layout.addWidget(deepseek_key_input)
        deepseek_group.setLayout(deepseek_layout)
        layout.addWidget(deepseek_group)
        
        # Instructions
        instructions = QLabel(
            "To use AI services, set the following environment variables:\n"
            "- ANTHROPIC_API_KEY for Claude\n"
            "- OPENAI_API_KEY for OpenAI\n"
            "- CHATGPT_API_KEY for ChatGPT\n"
            "- DEEPSEEK_API_KEY for DeepSeek\n\n"
            "You can set these in your shell or add them to your ~/.bashrc or ~/.zshrc file."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(settings_dialog.close)
        layout.addWidget(close_btn)
        
        settings_dialog.exec()

    def show_prompt_settings(self):
        dialog = PromptSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.save_prompts()
            self.load_prompts()  # Reload prompts after saving

    def load_prompts(self):
        try:
            with open("prompt_settings.json", "r") as f:
                self.prompts = json.load(f)
        except FileNotFoundError:
            # Use default prompts
            self.prompts = {
                "analysis": """Analyze this {language} code ({scope}) and provide detailed feedback:
1. {style_tag} style violations (highlight with [STYLE] tag)
2. Performance bottlenecks (highlight with [PERFORMANCE] tag)
3. Potential bugs (highlight with [BUG] tag)
4. Memory usage issues (highlight with [MEMORY] tag)
5. {compatibility_tag} (highlight with [COMPATIBILITY] tag)

Provide specific recommendations with line numbers where applicable.

{language} Code:
{code}""",
                "optimization": """Optimize this {language} code ({scope}) with:
1. {style_tag} style compliance
2. Performance improvements
3. Memory efficiency
4. {best_practices_tag}
5. Error handling improvements

Requirements:
- Provide the complete rewritten code
- Include explanations of key changes
- Format the response with clear section headers
- Focus on both performance and readability
- Ensure backward compatibility
- Maintain the original functionality

{language} Code:
{code}""",
                "improvement": """Improve this {language} code ({scope}) by implementing the following suggestions:
1. Fix {style_tag} style violations
2. Address performance bottlenecks
3. Fix potential bugs
4. Optimize memory usage
5. Ensure {compatibility_tag}

For each improvement:
- Show the original code
- Explain the issue
- Show the improved code
- Explain why the improvement is better

{language} Code:
{code}

Provide the complete improved code with explanations of key changes.
Format the response with clear section headers.""",
                "inline": """Improve this {language} code ({scope}) with inline comments only.
Focus on:
1. {style_tag} style compliance
2. Performance improvements
3. Memory efficiency
4. Error handling
5. Code readability

Requirements:
- Add clear, concise inline comments explaining key improvements
- Keep comments brief and focused on the improvement
- Use standard comment style for {language}
- Maintain the original code structure
- Do not add any explanations outside the code
- Do not include any section headers or additional text

Return ONLY the improved code with inline comments in a code block.

{language} Code:
{code}"""
            }

    def save_results(self):
        """Save the current analysis/optimization results to a file"""
        if not self.analysis_results:
            QMessageBox.warning(self, "Warning", "No results to save")
            return

        # Generate default filename based on current file and operation
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0] if self.current_file_path else "results"
        current_tab = self.output_tabs.currentIndex()
        
        # Determine operation type based on current tab
        if current_tab == 1:  # Analysis/Optimization Results
            operation = "analysis"
        elif current_tab == 2:  # Optimized Code
            operation = "optimized"
        elif current_tab == 3:  # Improved Code
            operation = "improved"
        elif current_tab == 4:  # Inline Improved Code
            operation = "inline_improved"
        else:
            operation = "results"
            
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{base_name}_{operation}_{timestamp}.txt"
        
        # Get the directory of the current file or use home directory
        default_dir = os.path.dirname(self.current_file_path) if self.current_file_path else os.path.expanduser("~")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            os.path.join(default_dir, default_filename),
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.analysis_results)
                self.update_status(f"Results saved to {file_path}", 0)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply stylesheet for a more modern look
    app.setStyle("Fusion")
    
    # Optional: Set a dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    window = CodeOptimizerApp()
    window.show()
    sys.exit(app.exec())
