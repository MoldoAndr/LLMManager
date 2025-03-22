# 🤖 Lightweight LLM API Server

A containerized API server for deploying and interacting with small-footprint LLMs (Large Language Models).

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10-green)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen)

## 📋 Overview

This project provides a lightweight, containerized API for running small-footprint LLMs that can be deployed on modest hardware. It supports multiple model types with conversation management through a RESTful API interface.

### Key Features

- 🔄 **Multi-model support**: Run and switch between different LLM implementations
- 💬 **Conversation management**: Maintain context across multiple interactions
- 🐳 **Containerized deployment**: Easy setup with Docker and docker-compose
- 📝 **Command-line interface**: Simple shell script for interacting with models
- 🔍 **Modular architecture**: Easily extend with additional model implementations
- 🏋️ **Resource efficient**: Optimized for running on limited hardware

## 🧠 Supported Models

The API currently supports the following LLM implementations:

- **Phi-2**: Microsoft's 2.7B parameter model using 4-bit quantization
- **TinyLlama**: The 1.1B parameter model running with llama.cpp
- **RWKV**: A RNN-based model with transformer-level performance

## 🚀 Quick Start

### Prerequisites

- Docker and docker-compose
- 1-4GB of RAM (depending on model)
- Models must be downloaded separately to the `models` directory

### Running with Docker

1. Start the server:
   ```bash
   docker-compose up -d
   ```

2. Verify the service is running:
   ```bash
   curl http://localhost:5000/health
   ```

## 💻 Using the API

### Command Line Interface

The included `llm_chat.sh` script provides an easy way to interact with the API:

```bash
# List available models
./llm_chat.sh list

# Create a new conversation
./llm_chat.sh create phi2 my_conversation

# Send a message
./llm_chat.sh chat my_conversation "What is the meaning of life?"

# View conversation history
./llm_chat.sh history my_conversation

# Reset a conversation
./llm_chat.sh reset my_conversation
```

### REST API Endpoints

- `GET /api/models`: List all available models
- `GET /api/conversations`: List all conversations
- `POST /api/conversation`: Create a new conversation
- `GET /api/conversation/<id>`: Get conversation history
- `POST /api/conversation/<id>/reset`: Reset a conversation
- `POST /api/chat`: Send a message to a conversation
- `GET /health`: Service health check

## 🔧 Configuration

### Docker Environment Variables

- `MODEL_DIR`: Directory for model files (default: `/app/models`)
- `DATA_DIR`: Directory for data files (default: `/app/data`)

### Model Initialization

Models are initialized at startup through the `initialize_models.sh` script. Modify this script to change which models are loaded by default.

## 🏗️ Architecture

The system consists of:

1. **LLMInterface**: Base class for all model implementations
2. **Model Implementations**: Phi2Model, TinyLlamaModel, RWKVModel
3. **LLMConversationManager**: Manages conversations and model interactions
4. **Flask API**: RESTful interface for client applications

## 🛠️ Development

### Adding New Models

To add support for a new model type:

1. Create a new class in `model_implementations.py` inheriting from `LLMInterface`
2. Implement the `generate()` and `get_model_size()` methods
3. Update the `load_model()` function in `app.py` to support the new model type

### Running Without Docker

1. Install Python 3.10 and required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```bash
   python app.py
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
