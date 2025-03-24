import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from plugins.database_plugin import DatabasePlugin
from plugins.storage_plugin import StoragePlugin
from plugins.query_plugin import QueryPlugin
from plugins.response_plugin import ResponsePlugin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Recon Agent API", description="Financial Reconciliation Agent built with Semantic Kernel")

# Initialize Semantic Kernel
async def initialize_kernel():
    # Create the kernel
    kernel = Kernel()
    
    # Configure Azure OpenAI
    azure_endpoint = os.environ.get("AZURE_OPENAI_API_BASE")
    azure_deployment = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    
    if not all([azure_endpoint, azure_deployment, api_key, api_version]):
        raise ValueError("Azure OpenAI configuration is incomplete. Check your .env file.")
    
    # Add Azure OpenAI service to the kernel
    azure_chat_service = AzureChatCompletion(
        deployment_name=azure_deployment,
        endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    kernel.add_service(azure_chat_service)
    
    # Import plugins
    kernel.import_plugin(DatabasePlugin(), plugin_name="database")
    kernel.import_plugin(StoragePlugin(), plugin_name="storage")
    kernel.import_plugin(QueryPlugin(), plugin_name="query")
    kernel.import_plugin(ResponsePlugin(), plugin_name="response")
    
    return kernel

# Startup event to initialize the kernel
@app.on_event("startup")
async def startup_event():
    app.state.kernel = await initialize_kernel()
    logger.info("Semantic Kernel initialized successfully")

# API endpoint to process queries
@app.post("/recon_agent")
async def recon_agent_endpoint(request: Request):
    try:
        # Get request data
        req_data = await request.json()
        user_input = req_data.get("chat_input")
        chat_history = req_data.get("chat_history", [])
        user_name = req_data.get("user_name", "")
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Please provide a valid question in the request body")
        
        logger.info(f"Received request from {user_name} with input: {user_input}")
        
        # Access the kernel
        kernel = app.state.kernel
        
        # Process the query
        # Step 1: Check query relevance
        variables = kernel.create_new_context()
        variables["user_input"] = user_input
        variables["chat_history"] = json.dumps(chat_history)
        variables["user_name"] = user_name
        
        relevance_result = await kernel.run_async(
            kernel.plugins["query"]["check_relevance"],
            input_vars=variables
        )
        relevance_data = json.loads(str(relevance_result))
        
        is_relevant = relevance_data.get("is_relevant", False)
        general_response = relevance_data.get("response", "")
        is_list_request = relevance_data.get("is_list_request", False)
        
        # Process based on query type
        if not is_relevant:
            logger.info(f"Query classified as non-relevant: {user_input}")
            return {"chat_output": general_response, "csv_url": None}
        
        if is_list_request:
            logger.info(f"Processing list request: {user_input}")
            # Execute list query
            list_result = await kernel.run_async(
                kernel.plugins["query"]["process_list_query"],
                input_vars=variables
            )
            list_data = json.loads(str(list_result))
            return {"chat_output": list_data.get("response", ""), "csv_url": list_data.get("csv_url")}
        else:
            logger.info(f"Processing regular financial query: {user_input}")
            # Generate SQL query
            sql_result = await kernel.run_async(
                kernel.plugins["query"]["generate_sql"],
                input_vars=variables
            )
            
            # Execute query
            variables["sql_query"] = str(sql_result)
            query_result = await kernel.run_async(
                kernel.plugins["database"]["execute_query"],
                input_vars=variables
            )
            
            # Check if we need to generate CSV
            variables["query_result"] = str(query_result)
            should_generate_csv = await kernel.run_async(
                kernel.plugins["query"]["should_generate_csv"],
                input_vars=variables
            )
            
            csv_url = None
            if str(should_generate_csv).lower() == "true":
                # Generate CSV and get URL
                csv_result = await kernel.run_async(
                    kernel.plugins["storage"]["generate_csv"],
                    input_vars=variables
                )
                csv_url = str(csv_result)
            
            # Format final response
            variables["csv_url"] = csv_url or ""
            response = await kernel.run_async(
                kernel.plugins["response"]["format_response"],
                input_vars=variables
            )
            
            return {"chat_output": str(response), "csv_url": csv_url}
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7071)