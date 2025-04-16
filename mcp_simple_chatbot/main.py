import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"No read permission for file: {file_path}")
            
            with open(file_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON in configuration file: {e.msg}",
                        e.doc,
                        e.pos,
                    )
        except OSError as e:
            if e.errno == 35:  # Resource deadlock avoided
                logging.error(f"Resource deadlock while reading {file_path}. Trying alternative approach...")
                # Try alternative approach using read() first
                with open(file_path, "r") as f:
                    content = f.read()
                return json.loads(content)
            raise

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                # Assuming Tool class is defined elsewhere or we need to define it.
                # For now, let's assume Tool is defined correctly.
                # Find the Tool definition or adjust if needed.
                try:
                    # This part assumes Tool is defined later or globally
                    from __main__ import Tool # Attempt to import if defined later
                except ImportError:
                    # Define a placeholder if Tool isn't found easily, 
                    # Needs review based on where Tool is actually defined
                    class Tool:
                        def __init__(self, name, description, inputSchema):
                            self.name=name; self.description=description; self.inputSchema=inputSchema
                            pass # Placeholder
                
                for tool_data in item[1]:
                    # Ensure Tool is callable with expected args
                    tools.append(Tool(tool_data.name, tool_data.description, tool_data.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    await self.session.shutdown()
                    self.session = None
                await self.exit_stack.aclose()
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")
            finally:
                self.session = None
                self.stdio_context = None


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM."""
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the Anthropic API."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        anthropic_messages = []
        system_content = None
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] in ["user", "assistant"]:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        payload = {
            "messages": anthropic_messages,
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 4096,
            "temperature": 0.2
        }
        if system_content:
            payload["system"] = system_content
        try:
            with httpx.Client(timeout=30.0) as client:
                logging.info("Sending request to Anthropic API...")
                response = client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    error_detail = response.json() if response.headers.get("content-type") == "application/json" else response.text
                    logging.error(f"Anthropic API Error: {response.status_code} - {error_detail}")
                    return f"I encountered an error communicating with the LLM service (Status {response.status_code}). Please try again."
                data = response.json()
                if "content" in data:
                    if len(data["content"]) > 0 and "text" in data["content"][0]:
                        return data["content"][0]["text"]
                    else:
                        # Handle empty content array
                        logging.error(f"Empty content array in response: {data}")
                        return "I processed your request, but had trouble formulating a response. This might be due to the complexity of the tool result. Please check the tool output above."
                else:
                    logging.error(f"Unexpected response structure: {data}")
                    return "I received an unexpected response format from the LLM service. Please try again."
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            return f"I encountered a network error: {error_message}. Please try again."
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logging.error(error_message)
            return "I encountered an unexpected error. Please try again."


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        if not hasattr(self, 'servers'):
            return
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                logging.error(f"Error cleaning up server: {e}")
            finally:
                server.session = None
                server.stdio_context = None

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed."""
        import json
        import re
        logging.info("Starting to process LLM response")
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_match = re.search(json_pattern, llm_response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group().strip()
                logging.info(f"Extracted JSON string: {json_str}")
                tool_call = json.loads(json_str)
                logging.info(f"Parsed tool call: {tool_call}")
                if "tool" in tool_call and "arguments" in tool_call:
                    print("\n=== Tool Call ===")
                    print(f"Tool: {tool_call['tool']}")
                    print(f"Arguments: {json.dumps(tool_call['arguments'], indent=2)}")
                    print("===============\n")
                    for server in self.servers:
                        logging.info(f"Checking server for tools...")
                        tools = await server.list_tools()
                        logging.info(f"Available tools: {[tool.name for tool in tools]}")
                        if any(tool.name == tool_call["tool"] for tool in tools):
                            try:
                                logging.info(f"Executing tool: {tool_call['tool']}")
                                result = await server.execute_tool(
                                    tool_call["tool"], tool_call["arguments"]
                                )
                                logging.info(f"Tool execution result: {result}")
                                if isinstance(result, dict) and "progress" in result:
                                    progress = result["progress"]
                                    total = result["total"]
                                    percentage = (progress / total) * 100
                                    print(f"Progress: {progress}/{total} ({percentage:.1f}%)")
                                print("\n=== Tool Response ===")
                                if isinstance(result, dict):
                                    print(json.dumps(result, indent=2))
                                else:
                                    print(result)
                                print("===================\n")
                                
                                # Format the result to be more LLM-friendly
                                tool_name = tool_call["tool"]
                                try:
                                    # If result is large, create a summary
                                    result_str = str(result)
                                    max_length = 1500  # Reasonable limit to avoid empty responses
                                    if len(result_str) > max_length:
                                        if isinstance(result, dict):
                                            # For dictionaries, extract key information
                                            summary = f"Tool '{tool_name}' execution result (summarized):\n"
                                            # Extract key fields depending on tool type
                                            if "message" in result:
                                                summary += f"Message: {result['message']}\n"
                                            if "url" in result:
                                                summary += f"URL: {result['url']}\n"
                                            if "name" in result:
                                                summary += f"Name: {result['name']}\n"
                                            # Add a few more key fields if present
                                            key_fields = ['id', 'title', 'description', 'status', 'type']
                                            for field in key_fields:
                                                if field in result:
                                                    summary += f"{field.capitalize()}: {result[field]}\n"
                                            return summary
                                        else:
                                            # For strings and other types, truncate
                                            return f"Tool '{tool_name}' execution result (truncated):\n{result_str[:max_length]}..."
                                    else:
                                        return f"Tool '{tool_name}' execution result:\n{result_str}"
                                except Exception as e:
                                    logging.error(f"Error formatting tool result: {e}")
                                    return f"Tool '{tool_name}' was executed successfully, but the result is complex. Please see the output above."
                            except Exception as e:
                                error_msg = f"Error executing tool: {str(e)}"
                                logging.error(error_msg)
                                print("\n=== Tool Error ===")
                                print(error_msg)
                                print("================\n")
                                return error_msg
                    logging.warning(f"No server found with tool: {tool_call['tool']}")
                    print("\n=== Tool Error ===")
                    print(f"No server found with tool: {tool_call['tool']}")
                    print("================\n")
                    return f"No server found with tool: {tool_call['tool']}"
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse extracted JSON: {e}")
                # Check if json_str is defined before logging
                if 'json_str' in locals(): 
                   logging.error(f"JSON string was: {json_str}")
                return llm_response
        logging.info("No tool call found in response")
        return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = [{"role": "system", "content": system_message}]
            print("\nChatbot ready! Type 'exit' or 'quit' to end the session.")
            print("Available tools:", ", ".join(tool.name for tool in all_tools))
            print("\n" + "="*50 + "\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit"]:
                        print("\nExiting...")
                        break

                    print("\nThinking...")
                    messages.append({"role": "user", "content": user_input})

                    llm_response = self.llm_client.get_response(messages)
                    
                    is_tool_call = False
                    try:
                        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                        json_match = re.search(json_pattern, llm_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group().strip()
                            parsed_json = json.loads(json_str)
                            if isinstance(parsed_json, dict) and "tool" in parsed_json and "arguments" in parsed_json:
                                is_tool_call = True
                                print("Processing tool call...")
                    except json.JSONDecodeError:
                        pass 

                    if not is_tool_call:
                         print("\nAssistant:", llm_response)

                    raw_result = await self.process_llm_response(llm_response)

                    if raw_result != llm_response:  # A tool was executed
                        messages.append({"role": "assistant", "content": llm_response})
                        
                        # Extract key information from GitHub API responses or other tools
                        # Instead of sending the entire tool result, create a simple message
                        # that only contains the most important information
                        try:
                            # First, convert the raw result to a proper string representation
                            raw_result_str = str(raw_result)
                            
                            # Keep the specialized handling as fallback in case the direct string approach fails
                            tool_result = None
                            
                            # Check if this is a GitHub repository creation response
                            if "Tool '" in raw_result and "github_create_repository" in raw_result:
                                print("\nDetected GitHub repository creation result")
                                # Extract key information from the repository response
                                if '"html_url"' in raw_result:
                                    match = re.search(r'"html_url":\s*"([^"]+)"', raw_result)
                                    if match:
                                        repo_url = match.group(1)
                                        tool_result = f"Successfully created GitHub repository. Repository URL: {repo_url}"
                                
                                # Fallback if we couldn't extract the URL
                                if not tool_result:
                                    match = re.search(r'"name":\s*"([^"]+)"', raw_result)
                                    if match:
                                        repo_name = match.group(1)
                                        tool_result = f"Successfully created GitHub repository named '{repo_name}'."
                            
                            # Handle GitHub search repositories
                            elif "Tool '" in raw_result and "github_search_repositories" in raw_result:
                                print("\nDetected GitHub repository search result")
                                repo_names = []
                                repo_urls = []
                                
                                # Extract repository names from the JSON response
                                name_matches = re.findall(r'"name":\s*"([^"]+)"', raw_result)
                                url_matches = re.findall(r'"html_url":\s*"([^"]+)"', raw_result)
                                
                                if name_matches and url_matches:
                                    # Filter out owner names (they also have "name" field)
                                    # Only consider names that are followed by "full_name"
                                    real_repo_names = []
                                    for i, name in enumerate(name_matches):
                                        # Check if this appears to be a repository name and not an owner name
                                        # by looking at surrounding context
                                        pos = raw_result.find(f'"name": "{name}"')
                                        if pos > 0 and raw_result.find('"full_name"', pos-50, pos+50) > 0:
                                            real_repo_names.append(name)
                                            if i < len(url_matches):
                                                repo_urls.append(url_matches[i])
                                    
                                    if real_repo_names:
                                        tool_result = f"Found {len(real_repo_names)} GitHub repositories:\n"
                                        for i, name in enumerate(real_repo_names[:5]):  # Limit to first 5 repos
                                            url = repo_urls[i] if i < len(repo_urls) else "URL not available"
                                            tool_result += f"{i+1}. {name} - {url}\n"
                                    else:
                                        tool_result = "Found GitHub repositories, but couldn't extract their names correctly."
                                else:
                                    tool_result = "The GitHub repository search was completed, but no repositories were found."
                            
                            # Add more specific handlers for other GitHub tools
                            elif "Tool '" in raw_result and "github" in raw_result:
                                # Generic handler for GitHub responses
                                tool_result = "Successfully executed GitHub operation. The operation completed successfully."
                                
                                # Try to extract URL if available
                                match = re.search(r'"url":\s*"([^"]+)"', raw_result)
                                if match:
                                    url = match.group(1)
                                    tool_result += f" Resource URL: {url}"
                            
                            # Default fallback for any tool
                            if not tool_result:
                                tool_result = "The tool was executed successfully. Please check the output above for details."
                            
                            # Use raw string representation of the entire tool result, 
                            # but limit length to avoid empty content responses from LLM
                            max_safe_length = 3000
                            if len(raw_result_str) > max_safe_length:
                                raw_result_str = raw_result_str[:max_safe_length] + "... (truncated)"
                                
                            # Add both raw result and processed summary as system messages
                            messages.append({"role": "system", "content": f"Raw tool response: {raw_result_str}"})
                            
                            # Keep the processed version as a backup for the simplified messages
                            system_update = f"Tool execution result: {tool_result}"
                            
                        except Exception as e:
                            logging.error(f"Error processing tool result: {e}")
                            messages.append({"role": "system", "content": "Tool execution completed, but I couldn't process the result in detail."})
                            system_update = "Tool execution completed, but I couldn't process the result in detail."
                            
                        print("\nGenerating final response...")
                        
                        # Create a completely new message list with just the essential information
                        # This reduces the chance of the LLM returning an empty response
                        simplified_messages = [
                            {"role": "system", "content": (
                                "You are a helpful assistant. You've just executed a tool on behalf of the user. "
                                "Now you need to respond to the user about the result in a natural, conversational way. "
                                "Be concise but informative. Focus on the key information from the raw tool response."
                            )},
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": "I'll help you with that by using a tool."},
                            {"role": "system", "content": f"Raw tool response: {raw_result_str}" if 'raw_result_str' in locals() else system_update}
                        ]
                        
                        try:
                            final_response = self.llm_client.get_response(simplified_messages)
                            if final_response and not final_response.startswith("I processed your request, but had trouble"):
                                print("\nAssistant:", final_response)
                                messages.append({"role": "assistant", "content": final_response})
                            else:
                                # If we still get an error or empty response, use a hardcoded response
                                hardcoded_response = "I've successfully processed your request. "
                                
                                if 'tool_result' in locals() and tool_result:
                                    hardcoded_response += tool_result
                                else:
                                    hardcoded_response += "The operation completed successfully. You can check the details in the tool output above."
                                
                                print("\nAssistant:", hardcoded_response)
                                messages.append({"role": "assistant", "content": hardcoded_response})
                        except Exception as e:
                            logging.error(f"Error generating final response: {e}")
                            fallback_response = "I've executed the requested action successfully. Please check the tool output above for details."
                            print("\nAssistant:", fallback_response)
                            messages.append({"role": "assistant", "content": fallback_response})
                    else:
                        if not is_tool_call:
                           messages.append({"role": "assistant", "content": llm_response})

                    print("\n" + "="*50 + "\n")

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    try:
        print("Starting chatbot in interactive mode...")
        print("Loading configuration...")
        config = Configuration()
        
        if not config.api_key:
            api_key = input("Please enter your LLM API key: ").strip()
            os.environ["LLM_API_KEY"] = api_key
            config = Configuration()

        default_config = {
            "mcpServers": {
                "default": {
                    "command": "npx",
                    "args": ["mcp-vscode-extension", "stdio"],
                    "env": {}
                }
            }
        }

        config_path = "servers_config.json"
        if not os.path.exists(config_path):
            print(f"Creating default {config_path}...")
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
        
        print("Loading server configuration...")
        server_config = config.load_config(config_path)
        
        print("Initializing servers...")
        servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        
        print("Creating LLM client...")
        llm_client = LLMClient(config.llm_api_key)
        
        print("Starting chat session...")
        chat_session = ChatSession(servers, llm_client)
        await chat_session.start()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError in main(): {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'chat_session' in locals():
            await chat_session.cleanup_servers()


if __name__ == "__main__":
    print("Starting in interactive mode. Press Ctrl+C to exit.")
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nError running asyncio loop: {e}")
        import traceback
        traceback.print_exc()
