import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from semantic_kernel.plugin_definition import kernel_function, KernelPlugin
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

class StoragePlugin(KernelPlugin):
    """Storage plugin for file operations and blob storage."""
    
    def __init__(self):
        """Initialize the storage plugin."""
        super().__init__()
    
    @kernel_function(
        description="Generate a CSV file from SQL query results and upload it to blob storage",
        name="generate_csv"
    )
    def generate_csv(self, sql_query: str, query_result: str) -> str:
        """
        Generate a CSV file from SQL query results and upload it to blob storage.
        
        Args:
            sql_query (str): The SQL query that was executed
            query_result (str): JSON string containing the query results
            
        Returns:
            str: URL of the uploaded blob or None if upload fails
        """
        try:
            # Parse the query result
            result_data = json.loads(query_result)
            
            if not result_data.get("success", False):
                return "null"
            
            column_names = result_data.get("column_names", [])
            rows = []
            
            # Convert the JSON data back to rows format
            for row_data in result_data.get("rows", []):
                row = [row_data.get(col, None) for col in column_names]
                rows.append(row)
            
            if not rows:
                return "null"
            
            # Generate a unique output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"/tmp/query_results_{timestamp}.csv"
            
            # Write to CSV
            blob_url = self._save_to_csv(column_names, rows, output_file)
            
            # Clean up the file after upload
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    logger.info(f"Temporary file {output_file} removed successfully")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {output_file}: {e}")
            
            return blob_url or "null"
            
        except Exception as e:
            logger.error(f"Error generating CSV: {e}")
            return "null"
    
    def _save_to_csv(self, column_names: List[str], rows: List[List[Any]], output_file: str) -> str:
        """
        Save the query results to a CSV file and upload to blob storage.
        
        Args:
            column_names (list): List of column names
            rows (list): List of data rows
            output_file (str): Path to save the CSV file
            
        Returns:
            str: Blob URL if uploaded successfully, otherwise None
        """
        try:
            # Write data to CSV file
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                writer.writerows(rows)
            logger.info(f"Results successfully saved to {output_file}")
            
            # Upload to blob storage
            blob_url = self._upload_to_blob_storage(output_file)
            return blob_url
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return None
    
    def _upload_to_blob_storage(self, file_path: str) -> str:
        """
        Upload a file to Azure Blob Storage
        
        Args:
            file_path (str): Path to the file to upload
            
        Returns:
            str: URL of the uploaded blob or None if upload fails
        """
        try:
            # Get blob storage details from environment variables
            connection_string = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
            container_name = os.getenv("BLOB_STORAGE_CONTAINER_NAME")
            account_url = os.getenv("BLOB_STORAGE_ACCOUNT_url")
            
            if not all([connection_string, container_name, account_url]):
                logger.error("Missing blob storage configuration")
                return None
                
            # Create blob service client
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            # Get container client
            container_client = blob_service_client.get_container_client(container_name)
            # Generate a unique blob name using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.basename(file_path)
            blob_name = f"{timestamp}_{file_name}"
            
            # Get blob client
            blob_client = container_client.get_blob_client(blob_name)
            
            # Upload file
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                
            # Generate the blob URL
            blob_url = f"{account_url}/{container_name}/{blob_name}"
            logger.info(f"File uploaded successfully to {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading file to blob storage: {e}")
            return None