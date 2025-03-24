import json
import logging
from typing import Dict
from semantic_kernel.plugin_definition import kernel_function, KernelPlugin
from openai import AzureOpenAI
import os

logger = logging.getLogger(__name__)

class ResponsePlugin(KernelPlugin):
    """Response plugin for formatting responses to the user."""
    
    def __init__(self):
        """Initialize the response plugin."""
        super().__init__()
        self.client = self._setup_openai_client()
        self.deployment_name = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT")
    
    def _setup_openai_client(self):
        """Setup OpenAI client for direct API calls."""
        try:
            client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
            )
            logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"OpenAI client initialization failed: {e}")
            raise ValueError(f"OpenAI client initialization failed: {e}")
    
    def _format_chat_history(self, chat_history_json: str) -> str:
        """Format chat history for prompt context."""
        try:
            chat_history = json.loads(chat_history_json)
            if not chat_history or len(chat_history) == 0:
                return "No previous conversation."
                
            formatted_history = ""
            for entry in chat_history:
                question = entry.get("question", "")
                answer = entry.get("answer", "")
                if question:
                    formatted_history += f"User: {question}\n"
                if answer:
                    formatted_history += f"Assistant: {answer}\n"
            return formatted_history
        except Exception as e:
            logger.error(f"Error formatting chat history: {e}")
            return "No previous conversation."
    
    @kernel_function(
        description="Format the response to the user based on query results",
        name="format_response"
    )
    def format_response(self, user_input: str, query_result: str, sql_query: str, chat_history: str, csv_url: str = "") -> str:
        """
        Format the response to the user based on query results.
        
        Args:
            user_input (str): The user's input query
            query_result (str): JSON string containing the query results
            sql_query (str): The SQL query that was executed
            chat_history (str): JSON string containing chat history
            csv_url (str): URL to the CSV file if available
            
        Returns:
            str: Formatted response to the user
        """
        try:
            formatted_chat_history = self._format_chat_history(chat_history)
            
            # Parse the query result
            result_data = json.loads(query_result)
            
            if not result_data.get("success", False):
                error_message = result_data.get("error", "Unknown error")
                return f"I encountered an error executing your query: {error_message}"
            
            rows = result_data.get("rows", [])
            column_names = result_data.get("column_names", [])
            
            # If no data found
            if not rows:
                return "No data found for the prompt."
            
            # Check if the result is too large
            if result_data.get("row_count", 0) > 10000:
                return "Too many records found for the prompt (exceeds 10000). Please refine your query."
            
            # Format the answer prompt
            answer_prompt = f"""
            You are a Basofa Financial Recon Agent. Given the following user question, corresponding SQL query, SQL result, and previous chat history, answer the user question based on the SQL result only and please do not generate any answer of your own.
            And If the SQL result is empty, then answer the user question as "No data found for the prompt." and please do not generate any hypothetical answer for user.
            You will handle tabular data as well. You will always include the emojis and appropriate symbols (like ✅, ✔️, ⬆️, 😊,👍,🙂,etc) and maintain user friendly format.
            If you receive error message about too many records, then show message to user as "Too many records found for the prompt (exceeds 1200). Please refine your query." and please do not provide any suggestion to user.
            
            IMPORTANT: If a CSV file has been generated (CSV Download URL is provided), ALWAYS include the exact text "Download URL: " followed by the URL at the end of your response. Do not format it as a Markdown link.
            
            Previous chat history: {formatted_chat_history}
            Question: {user_input}
            SQL Query: {sql_query}
            SQL Result: {json.dumps(rows[:50], default=str, indent=2)}  # Limiting to first 50 rows for token count
            Column Names: {column_names}
            Total Row Count: {result_data.get("row_count", 0)}
            CSV Download URL: {csv_url}
            Answer:  """
            
            # Generate the answer
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a Financial Reconciliation Agent that provides accurate and helpful information."},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=0
            )
            
            formatted_response = response.choices[0].message.content.strip()
            
            # If CSV URL exists but is not included in the response, add it
            if csv_url and "Download URL:" not in formatted_response:
                formatted_response += f"\n\nDownload URL: {csv_url}"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question. Error details: {str(e)}"