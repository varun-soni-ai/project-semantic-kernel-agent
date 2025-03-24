import os
import json
import logging
import pymssql
import sqlalchemy
from typing import List, Dict, Any, Tuple
from semantic_kernel.plugin_definition import kernel_function, KernelPlugin
from langchain.sql_database import SQLDatabase

logger = logging.getLogger(__name__)

class DatabasePlugin(KernelPlugin):
    """Database plugin for executing SQL queries against a SQL Server database."""
    
    def __init__(self):
        """Initialize the database plugin."""
        super().__init__()
        self.sql_db = self._setup_database()
    
    def _setup_database(self) -> SQLDatabase:
        """Setup database connection."""
        server = os.getenv("DB_server")
        port = os.getenv("Db_port")
        database = os.getenv("Database")
        username = os.getenv("DB_username")
        password = os.getenv("DB_password")
        
        connection_string = f"mssql+pymssql://{username}:{password}@{server}:{port}/{database}"
        
        try:
            sql_db = SQLDatabase.from_uri(
                connection_string,
                lazy_table_reflection=True,
                sample_rows_in_table_info=3,
                include_tables=['AdyenPaymentTransaction', 'BankPaymentTransaction']
            )
            logger.info("Database initialized successfully")
            return sql_db
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise ValueError(f"Database initialization failed: {e}")
    
    @kernel_function(
        description="Get the database schema information",
        name="get_schema"
    )
    def get_schema(self) -> str:
        """Get the database schema information."""
        try:
            return self.sql_db.table_info
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return f"Error getting database schema: {e}"
    
    @kernel_function(
        description="Execute a SQL query and return the results as JSON",
        name="execute_query"
    )
    def execute_query(self, sql_query: str) -> str:
        """
        Execute a SQL query and return the results as JSON.
        
        Args:
            sql_query (str): The SQL query to execute
            
        Returns:
            str: JSON string containing query results
        """
        try:
            # Clean up the query
            cleaned_query = sql_query.replace('```', '').replace('sql', '').replace("`","").strip()
            
            # Set up database connection using the environment parameters
            server = os.getenv("DB_server")
            port = os.getenv("Db_port")
            database = os.getenv("Database")
            username = os.getenv("DB_username")
            password = os.getenv("DB_password")
            
            # Create connection string for pymssql
            conn = pymssql.connect(
                server=server, 
                port=int(port) if port else None, 
                user=username, 
                password=password, 
                database=database
            )
            
            cursor = conn.cursor()
            cursor.execute(cleaned_query)
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            # Convert rows to JSON compatible format
            json_data = []
            for row in rows:
                json_row = {}
                for i, col_name in enumerate(column_names):
                    # Handle non-serializable types
                    value = row[i]
                    if isinstance(value, (set, complex)):
                        value = str(value)
                    json_row[col_name] = value
                json_data.append(json_row)
            
            result = {
                "success": True,
                "column_names": column_names,
                "rows": json_data,
                "row_count": len(rows)
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            error_result = {
                "success": False,
                "error": str(e)
            }
            return json.dumps(error_result)
    
    @kernel_function(
        description="Check if result size is too large for direct processing",
        name="check_result_size"
    )
    def check_result_size(self, result: str) -> str:
        """
        Check if the query result exceeds token limits.
        
        Args:
            result (str): JSON string with query results
            
        Returns:
            str: JSON with is_too_large flag and message
        """
        try:
            result_data = json.loads(result)
            row_count = result_data.get("row_count", 0)
            
            # Check if row count exceeds limit
            if row_count > 10000:
                return json.dumps({
                    "is_too_large": True,
                    "message": "Too many records found for the prompt (exceeds 10000 records). Please refine your query."
                })
            else:
                return json.dumps({
                    "is_too_large": False,
                    "message": "Result size is acceptable."
                })
                
        except Exception as e:
            logger.error(f"Error checking result size: {e}")
            return json.dumps({
                "is_too_large": True,
                "message": f"Error processing results: {str(e)}"
            })