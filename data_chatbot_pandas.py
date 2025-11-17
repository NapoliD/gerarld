"""
Generic CSV Data Analysis Chatbot using LangChain Pandas DataFrame Agent
Can analyze ANY CSV file by writing and executing pandas code directly.
"""

import os
import sys
from pathlib import Path
from typing import Dict
import pandas as pd

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PandasDataChatbot:
    """Interactive chatbot for CSV data analysis using pandas agent."""

    def __init__(self, data_dir: str = 'datos', api_key: str = None):
        """Initialize the chatbot."""
        print("\n" + "="*80)
        print("CSV DATA ANALYSIS CHATBOT (Pandas Agent)")
        print("="*80)

        self.data_dir = Path(data_dir)

        # Check data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Count CSV files
        self.csv_files = {f.name: f for f in self.data_dir.glob("*.csv")}
        print(f"\n[1/3] Data directory: {data_dir}")
        print(f"      Found {len(self.csv_files)} CSV files:")
        for i, filename in enumerate(self.csv_files.keys(), 1):
            size_mb = self.csv_files[filename].stat().st_size / (1024 * 1024)
            print(f"        {i}. {filename} ({size_mb:.2f} MB)")

        # Setup Gemini
        print(f"\n[2/3] Connecting to Gemini AI...")
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
            temperature=0  # Deterministic for code generation
        )
        print("[OK] Connected to Gemini Pro")

        # Store loaded dataframes
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.current_agent = None
        self.current_file = None

        print(f"\n[3/3] Chatbot initialization complete!")
        print("="*80)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame."""
        if filename not in self.csv_files:
            # Try to find partial match
            matches = [f for f in self.csv_files.keys() if filename.lower() in f.lower()]
            if matches:
                filename = matches[0]
                print(f"[INFO] Using closest match: {filename}")
            else:
                raise ValueError(
                    f"CSV file '{filename}' not found.\n"
                    f"Available files: {', '.join(self.csv_files.keys())}"
                )

        # Check if already loaded
        if filename in self.dataframes:
            print(f"[INFO] Using cached dataframe for {filename}")
            return self.dataframes[filename]

        # Load CSV
        file_path = self.csv_files[filename]
        print(f"[INFO] Loading {filename}...")

        try:
            df = pd.read_csv(file_path)
            self.dataframes[filename] = df
            print(f"[OK] Loaded {len(df):,} rows x {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {str(e)}")

    def analyze(self, filename: str, question: str) -> str:
        """Analyze a specific CSV file with a question."""
        # Load the dataframe
        df = self.load_csv(filename)

        # Create or update pandas agent
        if self.current_file != filename or self.current_agent is None:
            print(f"[INFO] Creating pandas agent for {filename}...")
            self.current_agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                allow_dangerous_code=True,  # Required for pandas agent
                agent_type="zero-shot-react-description",  # Compatible with Gemini
                prefix=f"""You are an expert data analyst working with a CSV file called '{filename}'.

The dataframe is already loaded and available as 'df'.

Your capabilities:
1. Analyze data structure and statistics
2. Check data quality (missing values, duplicates, outliers)
3. Perform exploratory data analysis (distributions, correlations)
4. Suggest data improvements and transformations
5. Write and execute pandas code to answer questions

When analyzing data:
- Use df.info(), df.describe(), df.isnull().sum() for overview
- Check for outliers using IQR method or statistical methods
- Look for patterns, correlations, and anomalies
- Provide actionable recommendations with code examples
- Be specific about data quality issues

Always explain your findings clearly and provide code examples when suggesting improvements.
"""
            )
            self.current_file = filename
            print(f"[OK] Agent ready for {filename}")

        # Get response
        try:
            response = self.current_agent.invoke({"input": question})
            return response['output']
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def chat(self, message: str) -> str:
        """General chat interface - routes to appropriate handler."""
        message_lower = message.lower()

        # Try to extract filename from message FIRST (before checking list commands)
        filename = None
        for csv_name in self.csv_files.keys():
            if csv_name.lower() in message_lower:
                filename = csv_name
                break

        # If filename found, analyze it
        if filename:
            # Remove filename from question to get the actual query
            question = message.replace(filename, '').strip()
            # Remove common prefixes
            for prefix in ['analyze', 'check', 'quality of', 'show me', 'what are', 'what is']:
                if question.lower().startswith(prefix):
                    question = question[len(prefix):].strip()

            # If question is empty or too short, provide comprehensive analysis
            if not question or len(question) < 10:
                question = "Provide a comprehensive analysis of this dataset including structure, statistics, data quality issues, and improvement suggestions."

            return self.analyze(filename, question)

        # List files command (only if no filename detected)
        if any(keyword in message_lower for keyword in ['list files', 'show files', 'what files', 'what csv', 'available files', 'which files']):
            result = f"\nAVAILABLE CSV FILES ({len(self.csv_files)}):\n"
            result += "="*80 + "\n\n"
            for i, (filename, filepath) in enumerate(self.csv_files.items(), 1):
                size_mb = filepath.stat().st_size / (1024 * 1024)
                result += f"{i}. {filename} ({size_mb:.2f} MB)\n"
            result += "\n" + "="*80
            result += "\n\nTo analyze a file, ask: 'Analyze <filename>' or 'Check quality of <filename>'"
            return result

        # No filename detected and no list command
        return (
            "I need to know which CSV file to analyze.\n\n"
            "Available files:\n" +
            "\n".join([f"  - {name}" for name in self.csv_files.keys()]) +
            "\n\nPlease specify a filename in your question, for example:\n"
            "  - 'Analyze olist_orders_dataset.csv'\n"
            "  - 'Check quality of products.csv'\n"
            "  - 'What are the missing values in customers.csv?'"
        )

    def run_interactive(self):
        """Run interactive chat session."""
        print("\n" + "="*80)
        print("CHATBOT READY - Ask about your CSV data!")
        print("="*80)
        print("\nExamples of what you can ask:")
        print("  - What CSV files are available?")
        print("  - Analyze olist_orders_dataset.csv")
        print("  - Check data quality of olist_products_dataset.csv")
        print("  - What are the missing values in olist_customers_dataset.csv?")
        print("  - Show me correlations in olist_order_items_dataset.csv")
        print("  - Suggest improvements for olist_reviews_dataset.csv")
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

    parser = argparse.ArgumentParser(description='CSV Data Analysis Chatbot with Pandas Agent')
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\napol\OneDrive\Escritorio\consultora_data_smart\Gerarld\datos',
                        help='Directory containing CSV files')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Google API key (or set GOOGLE_API_KEY env variable)')

    args = parser.parse_args()

    # Initialize chatbot
    try:
        chatbot = PandasDataChatbot(
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
        print("4. Install required packages: pip install langchain-experimental")
        sys.exit(1)


if __name__ == '__main__':
    main()
