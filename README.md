# Code Optimizer with AI
A multi-language code optimizer application with AI integration

## Screenshot
![Code Optimizer Interface](Screenshot.png)

## Features
- Multi-language code optimization
- AI-powered code suggestions
- Support for multiple AI models (Ollama, Claude, OpenAI)
- Real-time progress updates
- Code comparison view

## Code Structure
The application is built using PyQt6 and consists of several key components:

### Main Components
- `CodeOptimizerApp`: The main application class that handles the GUI and user interactions
- `AIWorker`: A QThread-based worker that handles AI model interactions asynchronously
- `CodeEditor`: A custom QTextEdit widget with syntax highlighting support
- `SettingsDialog`: A dialog for managing API keys and other settings

### Key Features Implementation
- **Multi-language Support**: Uses language-specific syntax highlighting and optimization prompts
- **AI Integration**: Supports multiple AI models through a unified interface
- **Real-time Updates**: Implements Qt signals and slots for asynchronous processing
- **Code Comparison**: Uses a split view to show original and optimized code side by side

### Dependencies
- PyQt6: For the graphical user interface
- OpenAI: For GPT model integration
- Anthropic: For Claude model integration
- Transformers: For local model support
