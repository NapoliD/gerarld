"""
Generic CSV Data Analysis Chatbot with LangChain + Gemini
Analyzes ANY CSV file, performs EDA, checks quality, and suggests improvements.
"""

import os
import sys
from pathlib import Path
from typing import List

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports
from src.generic_data_tools import get_generic_data_tools


# Agent prompt template
AGENT_PROMPT = """You are an expert data analyst specializing in data quality and exploratory data analysis.
You can analyze ANY CSV file and provide insights about:
1. Data structure and types
2. Data quality issues
3. Statistical distributions
4. Missing values and outliers
5. Improvement recommendations

Your goal is to help users understand their data and prepare it for analysis or machine learning.

IMPORTANT GUIDELINES:
- First, always list available CSV files to see what data exists
- Inspect structure before doing deep analysis
- Be specific about data quality issues
- Provide actionable, code-ready recommendations
- Focus on practical improvements that matter
- Explain WHY each improvement is needed

You have access to these tools:
{tools}

Tool Names: {tool_names}

HOW TO USE TOOLS:
1. list_csv_files - See all available CSV files
2. inspect_csv_structure - Check structure of a specific file
3. analyze_csv_data - Deep exploratory analysis
4. check_data_quality - Find quality issues
5. suggest_data_improvements - Get improvement recommendations

When you need to use a tool, follow this format:

Thought: Think about what information you need
Action: the tool name (must be one of {tool_names})
Action Input: the input for the tool
Observation: the result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final response to the user

Previous conversation:
{chat_history}

Current question: {input}

{agent_scratchpad}"""


class DataAnalysisChatbot:
    """Interactive chatbot for CSV data analysis."""

    def __init__(self, data_dir: str = 'datos', api_key: str = None):
        """Initialize the chatbot."""
        print("\n" + "="*80)
        print("INITIALIZING CSV DATA ANALYSIS CHATBOT")
        print("="*80)

        self.data_dir = data_dir

        # Check data directory
        if not Path(data_dir).exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Count CSV files
        csv_count = len(list(Path(data_dir).glob("*.csv")))
        print(f"\n[1/4] Data directory: {data_dir}")
        print(f"      Found {csv_count} CSV files")

        # Setup tools
        print("\n[2/4] Setting up data analysis tools...")
        self.tools = get_generic_data_tools(data_dir=data_dir)
        print(f"[OK] {len(self.tools)} tools ready:")
        for tool in self.tools:
            print(f"  - {tool.name}")

        # Setup Gemini
        print("\n[3/4] Connecting to Gemini AI...")
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError(
                "Google API Key not found!\n"
                "Please set GOOGLE_API_KEY environment variable or pass api_key parameter.\n"
                "Get your free API key at: https://makersuite.google.com/app/apikey"
            )

        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",  # Using Gemini 2.5 Flash (stable)
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        print("[OK] Connected to Gemini Pro")

        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            output_key="output"
        )

        # Create agent
        print("\n[4/4] Creating conversational agent...")
        prompt = PromptTemplate(
            template=AGENT_PROMPT,
            input_variables=["input", "agent_scratchpad", "chat_history"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

        print("[OK] Agent ready!\n")
        print("="*80)
        print("CHATBOT READY - Ask about your CSV data!")
        print("="*80)

    def chat(self, message: str) -> str:
        """Send a message to the chatbot and get response."""
        try:
            response = self.agent_executor.invoke({"input": message})
            return response['output']
        except Exception as e:
            return f"Error: {str(e)}"

    def run_interactive(self):
        """Run interactive chat session."""
        print("\nExamples of what you can ask:")
        print("  - What CSV files are available?")
        print("  - Analyze olist_orders_dataset.csv")
        print("  - Check data quality of products.csv")
        print("  - Suggest improvements for customers.csv")
        print("  - Show me the structure of reviews.csv")
        print()

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Check for exit
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nChatbot: Goodbye! Thanks for using the Data Analysis Chatbot.")
                    break

                # Get response
                print("\nChatbot: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nChatbot: Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CSV Data Analysis Chatbot')
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\napol\OneDrive\Escritorio\consultora_data_smart\Gerarld\datos',
                        help='Directory containing CSV files')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Google API key (or set GOOGLE_API_KEY env variable)')

    args = parser.parse_args()

    # Initialize chatbot
    try:
        chatbot = DataAnalysisChatbot(
            data_dir=args.data_dir,
            api_key=args.api_key
        )

        # Run interactive session
        chatbot.run_interactive()

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize chatbot: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set GOOGLE_API_KEY environment variable")
        print("2. Get a free API key at: https://makersuite.google.com/app/apikey")
        print("3. Check that the data directory exists and contains CSV files")
        sys.exit(1)


if __name__ == '__main__':
    main()
