# MCP Simple Chatbot

A modular chatbot interface leveraging Claude 3.7 Sonnet via the Anthropic API, with support for executing external tools through the MCP (Machine Conversation Protocol) standard.

## Features

- Interactive CLI chatbot interface
- Integration with the Anthropic Claude 3.7 Sonnet model
- Support for executing external tools via MCP
- Built-in GitHub API tool integration
- Automatic retry mechanism for tool execution
- Error handling and graceful degradation
- Docker deployment support

## Prerequisites

- Python 3.9+
- Node.js and NPX (for MCP tools)
- Docker (optional, for containerized deployment)
- Anthropic API key
- Github API key

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mcp-simple-chatbot.git
   cd mcp-simple-chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```


3. Create a `.env` file in the project root and add your Anthropic API key:
   ```
   LLM_API_KEY=your_anthropic_api_key_here
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t mcp-chatbot .
   ```

2. Run the container:
   ```bash
   docker run -it --env-file .env mcp-chatbot
   ```

## Configuration

The chatbot uses a configuration file (`servers_config.json`) to define available MCP servers and tools. A default configuration is created automatically on first run, but you can customize it in server_config.json:

```json
{
  "mcpServers": {
    "default": {
      "command": "npx",
      "args": ["mcp-vscode-extension", "stdio"],
      "env": {}
    }
  }
}
```

## Usage

### Running the Chatbot

Start the chatbot in interactive mode:

```bash
python mcp_simple_chatbot/main.py
```

### Interacting with the Chatbot

Once started, you can interact with the chatbot by typing messages. To execute tools, simply ask a question that requires a tool (like GitHub operations).

Example interactions:

- **Regular question**: "What's the weather like today?"
- **Tool usage**: "Search for repositories about machine learning on GitHub"
- **Quit**: Type "exit" or "quit" to end the session

### Available Tools

The chatbot will display available tools when started. These typically include:

- GitHub API tools (search repositories, create repositories, etc.)
- Other MCP-compatible tools

## Environment Variables

- `LLM_API_KEY`: Your Anthropic API key (required)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
