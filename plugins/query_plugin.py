import os
import json
import logging
from typing import Dict
from semantic_kernel.plugin_definition import kernel_function, KernelPlugin, kernel_function_context_parameter
from semantic_kernel import Kernel, KernelContext
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

class QueryPlugin(KernelPlugin):
    """Query plugin for classifying and generating SQL queries."""
    
    def __init__(self):
        """Initialize the query plugin."""
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
    
    def _extract_last_interaction(self, chat_history_json: str):
        """Extract the last meaningful interaction from chat history."""
        try:
            chat_history = json.loads(chat_history_json)
            if not chat_history or len(chat_history) == 0:
                return None
            
            # Get the last entry with both question and answer
            last_interaction = None
            for entry in reversed(chat_history):
                question = entry.get("question", "").strip()
                answer = entry.get("answer", "").strip()
                if question and answer:
                    last_interaction = {
                        "question": question,
                        "answer": answer
                    }
                    break
                    
            return last_interaction
        except Exception as e:
            logger.error(f"Error extracting last interaction: {e}")
            return None
    
    @kernel_function(
        description="Check if the user query is relevant to financial data, a general greeting, or requesting a list of transactions",
        name="check_relevance"
    )
    def check_relevance(self, user_input: str, chat_history: str, user_name: str = "") -> str:
        """
        Check if the user query is relevant to financial data, a general greeting, or requesting a list of transactions.
        
        Args:
            user_input (str): The user's input query
            chat_history (str): JSON string containing chat history
            user_name (str): Name of the user if available
            
        Returns:
            str: JSON string with classification results
        """
        try:
            formatted_chat_history = self._format_chat_history(chat_history)
            last_interaction = self._extract_last_interaction(chat_history)
            
            # Prepare user name for greeting
            greeting_name = f", {user_name}" if user_name and user_name.strip() else ""
            
            prompt = f"""You are a Financial Reconciliation Agent that analyzes transaction data.
                        You need to determine if the user's query is a general greeting, a data-related question, or specifically a request for a list of transactions.
                        CONTEXT RULES:
                            1. If the user mentions specific financial terms, database entities, or asks about data, ALWAYS classify as RELEVANT.
                            2. If the user refers to previous financial queries from chat history, ALWAYS classify as RELEVANT.
                            3. If the query mentions any specific store number, transaction number, reference number, or other entity IDs from previous chat, classify as RELEVANT.
                            4. If the user is clearly asking for a financial result or follow-up to previous data query, classify as RELEVANT.
                            5. Basic greetings with no financial context should be classified as NOT RELEVANT.
                            6. If the query is like summrise the data for some time period, classify as LIST_REQUEST = false and for all summarised data, classify as LIST_REQUEST = false.
                            7. If the user explicitly asks to "list", "show all", "display all" transactions, classify as LIST_REQUEST = true.
                            8. If the query contains terms like "list", "give me all", "all transactions", "display transactions", classify as LIST_REQUEST = true.
                            9. If the query asks for data for a specific time period (like "April", "last month", "yesterday") AND requests multiple transactions, classify as LIST_REQUEST = true.
                            10. If the query is asking for specific PSPreferences or requesting a set of transactions, classify as LIST_REQUEST = true.
                            11. If the query is ambiguous but seems to want comprehensive transaction data, classify as LIST_REQUEST = false.
                        Previous chat history:
                        {formatted_chat_history}
                        User query: "{user_input}"
                        Output format: Provide a JSON response with these fields:
                        - "is_relevant": true or false
                        - "response": If not relevant, provide a personalized greeting that references their last query if available
                        - "refers_to_previous": true if the query appears to reference previous data or questions
                        - "is_list_request": true if the query appears to be asking for a list of transactions
                        - "list_reasoning": brief explanation of why this is or isn't a list request
                        JSON response:"""
                        
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a classifier for a financial agent."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            # Extract and parse the JSON response
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            
            logger.info(f"Relevance check result: {result}")
            
            # If it's not relevant but we have chat history, create a personalized greeting
            if not result["is_relevant"] and last_interaction:
                # Format a personalized greeting that references their last query
                custom_greeting = f"""Hi{greeting_name}! I'm your Financial Reconciliation Agent. In our previous conversation, you asked about {last_interaction['question']}
                                      How can I help you with your financial data analysis today?"""
                result["response"] = custom_greeting
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error in relevance check: {str(e)}")
            # Default to treating the query as relevant in case of errors
            return json.dumps({
                "is_relevant": True,
                "response": "",
                "is_list_request": False
            })
    
    @kernel_function(
        description="Generate a SQL query based on the user input",
        name="generate_sql"
    )
    def generate_sql(self, user_input: str, chat_history: str) -> str:
        """
        Generate a SQL query based on the user input.
        
        Args:
            user_input (str): The user's input query
            chat_history (str): JSON string containing chat history
            
        Returns:
            str: Generated SQL query
        """
        try:
            formatted_chat_history = self._format_chat_history(chat_history)
            
            # First, rephrase the question to better suit our database schema
            rephrase_prompt = f'''You are tasked to rephrase questions according to this database. You will always rephrase user questions to get summaries since there is a very large database. Your output will be provided to a SQL agent that converts natural language to SQL queries based on the database info.
                                 Key principles for rephrasing:
                                    1. ALWAYS preserve specific filters mentioned in the original question:
                                    - Payment status/types (e.g., Captured, Authorized, Refused)
                                    - Payment methods (e.g., Visa, Mastercard)
                                    - Date ranges
                                    - Amount thresholds
                                    - Store numbers
                                    - Channel names

                                    2. For reconciliation requests:
                                    - Maintain ALL original filters in the rephrased version
                                    - Compare between Adyen and Bank systems
                                    - Get summaries for the data like how many transactions are matching and how many are not matching.
                                    - Include key comparison points:
                                        * Amount comparisons (PAYMENTAMOUNT vs CAPTUREDAMOUNT)
                                        * Status matching (PAYMENTSTATUS vs TRANSACTIONTYPE)
                                        * Date alignment (TRANSACTIONDATETIME from both systems)
                                        * Reference matching (PSPREFERENCE)
                                        * Missing transactions in either system

                                    3. Structure of rephrased questions:
                                    - Start with "Provide summary of" or "Summarize"
                                    - Include original date ranges exactly as specified
                                    - Keep all original filters and conditions
                                    - Add reconciliation aspects if comparing systems
                                    
                                    Database info: CREATE TABLE [AdyenPaymentTransaction] (
                                                    [PSPREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL, 
                                                    [MERCHANTREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [TRANSACTIONDATETIME] DATETIME NULL, 
                                                    [TIMEZONE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [CURRENCY] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTMETHOD] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTSTATUS] NVARCHAR(20) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [RISKSCORE] INTEGER NULL, 
                                                    CONSTRAINT [PK__AdyenPay__167A345A1489B7D8] PRIMARY KEY ([PSPREFERENCE])
                                                )

                                                /*
                                                3 rows from AdyenPaymentTransaction table:
                                                PSPREFERENCE	MERCHANTREFERENCE	TRANSACTIONDATETIME	TIMEZONE	PAYMENTAMOUNT	CURRENCY	PAYMENTMETHOD	PAYMENTSTATUS	RISKSCORE
                                                B25KD5G9X5STZK65	cfdcccacc90d4a84a20d313c680133fa	2024-04-16 17:57:00	IST	185.94	USD	unknown card	Refused	0
                                                B3J7GSMB4FVJK8V5	c33ad9b319544586b2cb9def832aa6f5	2024-07-08 10:35:00	IST	17.25	USD	Visa	Cancelled	1
                                                B4WPLLG2DTPRCK65	3a97f78e98714365a402f36d12ecc5cb	2024-07-12 11:45:00	IST	66.44	USD	Visa	Cancelled	1
                                                */
                                                CREATE TABLE [BankPaymentTransaction] (
                                                    [STORENUMBER] INTEGER NULL, 
                                                    [CHANNELNAME] VARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [LAST4DIGITS] CHAR(4) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [CARDID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [TRANSACTIONNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [TRANSACTIONDATETIME] DATETIME NULL, 
                                                    [CAPTUREDAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [TRANSACTIONTYPE] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PSPREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL, 
                                                    [PAYMENTMETHOD] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [SETTLEMENTDATE] DATE NULL, 
                                                    [SETTLEMENTID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [MERCHANTID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [GROSSAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [TRANSACTIONFEES] DECIMAL(10, 2) NULL, 
                                                    [NETAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [CURRENCY] NVARCHAR(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [DEPOSITAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [DEPOSITDATE] DATE NULL, 
                                                    [BANKACCOUNTNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [REFERENCENUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTPROVIDER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [BATCHNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    CONSTRAINT [PK__BankPaym__2CABD762F33B242E] PRIMARY KEY ([PSPREFERENCE])
                                                )

                                                /*
                                                3 rows from BankPaymentTransaction table:
                                                STORENUMBER	CHANNELNAME	LAST4DIGITS	CARDID	TRANSACTIONNUMBER	TRANSACTIONDATETIME	CAPTUREDAMOUNT	TRANSACTIONTYPE	PSPREFERENCE	PAYMENTMETHOD	SETTLEMENTDATE	SETTLEMENTID	MERCHANTID	GROSSAMOUNT	TRANSACTIONFEES	NETAMOUNT	CURRENCY	DEPOSITAMOUNT	DEPOSITDATE	BANKACCOUNTNUMBER	REFERENCENUMBER	PAYMENTPROVIDER	BATCHNUMBER
                                                5425	Fabrikam call center	1111	VISA	d3271376228543029d0bd99f9a9224a4	2024-07-17 06:44:59	28.82	Authorize	BMXXCC9MFC3GDXT5	CreditCard	2024-10-14	1234	b0183b8f0bec4bd2a47c52b1ecebe8b10	28.82	2.00	26.82	USD	28.82	2024-10-14	2345678901	b0183b8f0bec12	Adyen 	V6ZMHFVSV22WQT22
                                                5413	Fabrikam call center	1111	VISA	090178b06d65458e87e67e66092e63d8	2024-07-17 06:59:49	28.82	Capture	D44GVXL969NKGK82	CreditCard	2024-10-14	1234	b0183b8f0bec4bd2a47c52b1ecebe8b2	28.82	2.00	26.82	USD	28.82	2024-10-14	2345678901	b0183b8f0bec4	Adyen 	V6ZMHFVSV22WQT10
                                                5415	Fabrikam call center	1111	VISA	090178b06d65458e87e67e66092e63d8s	2024-07-17 06:59:49	38.99	Capture	D44GVXL969NKGK84	CreditCard	2024-10-14	1234	b0183b8f0bec4bd2a47c52b1ecebe8b3	38.99	2.00	36.99	USD	38.99	2024-10-14	2345678901	b0183b8f0bec5	Adyen 	V6ZMHFVSV22WQT12
                                                */

                                    Previous chat history: {formatted_chat_history}
                                    Original question: {user_input}
                                    Rephrased question:'''
                                    
            # Get rephrased question
            rephrase_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at rephrasing questions for SQL databases."},
                    {"role": "user", "content": rephrase_prompt}
                ],
                temperature=0
            )
            
            rephrased_question = rephrase_response.choices[0].message.content.strip()
            
            # Now generate the SQL query based on the rephrased question
            sql_prompt = f'''
            You are a Microsoft SQL Server (SSMS) expert. Given a question, generate a precise SQL query with these guidelines:
            1. Always retrieve relevant columns and columns that provides some specifics about the data- never use SELECT *
            2. Use appropriate JOIN (like LEFT JOIN, Inner join, Right Join, Outer join) operations when data spans multiple tables.
            3. There are different number of columns in both tables. 
            4. Handle date ranges using proper date functions and formats
            5. For summary requests (indicated by "Provide", "Give", or "Summarize"):
            - First attempt to query from a single table
            - Use JOINs only if required data spans multiple tables
            - Include aggregation functions as needed (COUNT, SUM, AVG etc.)

            Key constraints:
            - No DML/DDL queries
            - Query only existing columns from provided table schemas
            - Use CAST(GETDATE() as date) for current date references
            - Please do not always use 600 results for all question that ask to provide single data point or specific data for PSPREFERENCE, TRANSACTIONNUMBER, MERCHANTID, STORENUMBER, and MERCHANTREFERENCE
            - Please only use 600 results for questions that ask to provide/give the list of transactions
            - Ensure proper handling of NULL values and data types
            - Use explicit column names in GROUP BY and ORDER BY clauses
            - Please only provide SQL query with no explanations

            Only use the following tables and deeply understand the tables schema and generated the query accordingly. Always retrieve all relevant columns from table. And correct the query if the user request is not in align with the table schema. Here is some information about the tables: Distinct TRANSACTIONTYPE in BankPaymentTransaction: Authorize, Void, Capture
            Distinct PAYMENTSTATUS in AdyenPaymentTransaction: Refused, Settled, Cancelled, SettledExternally, Authorised
            
            Previous chat history: {formatted_chat_history}
            Question: {rephrased_question}
            Here is the table schema:
            CREATE TABLE [AdyenPaymentTransaction] (
                [PSPREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL, 
                [MERCHANTREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [TRANSACTIONDATETIME] DATETIME NULL, 
                [TIMEZONE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [PAYMENTAMOUNT] DECIMAL(10, 2) NULL, 
                [CURRENCY] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [PAYMENTMETHOD] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [PAYMENTSTATUS] NVARCHAR(20) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [RISKSCORE] INTEGER NULL, 
                CONSTRAINT [PK__AdyenPay__167A345A1489B7D8] PRIMARY KEY ([PSPREFERENCE])
            )
            
            CREATE TABLE [BankPaymentTransaction] (
                [STORENUMBER] INTEGER NULL, 
                [CHANNELNAME] VARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [LAST4DIGITS] CHAR(4) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [CARDID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [TRANSACTIONNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [TRANSACTIONDATETIME] DATETIME NULL, 
                [CAPTUREDAMOUNT] DECIMAL(10, 2) NULL, 
                [TRANSACTIONTYPE] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [PSPREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL, 
                [PAYMENTMETHOD] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [SETTLEMENTDATE] DATE NULL, 
                [SETTLEMENTID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [MERCHANTID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [GROSSAMOUNT] DECIMAL(10, 2) NULL, 
                [TRANSACTIONFEES] DECIMAL(10, 2) NULL, 
                [NETAMOUNT] DECIMAL(10, 2) NULL, 
                [CURRENCY] NVARCHAR(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [DEPOSITAMOUNT] DECIMAL(10, 2) NULL, 
                [DEPOSITDATE] DATE NULL, 
                [BANKACCOUNTNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [REFERENCENUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [PAYMENTPROVIDER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                [BATCHNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                CONSTRAINT [PK__BankPaym__2CABD762F33B242E] PRIMARY KEY ([PSPREFERENCE])
            )
            
            SQLQuery:
            '''
            
            sql_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator."},
                    {"role": "user", "content": sql_prompt}
                ],
                temperature=0
            )
            
            sql_query = sql_response.choices[0].message.content.strip()
            # Clean up the query
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return "SELECT TOP 10 * FROM AdyenPaymentTransaction; -- Error generating specific query"
    
    @kernel_function(
        description="Process a list query that's specifically requesting a list of transactions",
        name="process_list_query"
    )
    def process_list_query(self, user_input: str, chat_history: str) -> str:
        """
        Process a query that's specifically requesting a list of transactions.
        
        Args:
            user_input (str): The user's query asking for a list
            chat_history (str): JSON string containing chat history
            
        Returns:
            str: JSON string with the response and CSV URL
        """
        try:
            formatted_chat_history = self._format_chat_history(chat_history)
            
            # Generate the SQL query based on the user input
            query_generation_prompt = f"""
            You are a Microsoft SQL Server expert. Generate a precise SQL query for retrieving a list of transactions based on this request:
            
            "{user_input}"
            
            Follow these guidelines:
            1. Include ALL relevant columns that would be needed to understand each transaction
            2. For AdyenPaymentTransaction include: PSPREFERENCE, MERCHANTREFERENCE, TRANSACTIONDATETIME, PAYMENTAMOUNT, CURRENCY, PAYMENTMETHOD, PAYMENTSTATUS, RISKSCORE
            3. For BankPaymentTransaction include: STORENUMBER, CHANNELNAME, TRANSACTIONNUMBER, TRANSACTIONDATETIME, CAPTUREDAMOUNT, TRANSACTIONTYPE, PSPREFERENCE, PAYMENTMETHOD, SETTLEMENTDATE
            4. Use appropriate JOIN operations if the request spans both tables
            5. Apply proper filtering based on the request (dates, status, amount, etc.)
            6. DO NOT use aggregations like SUM, COUNT, AVG in this query - we need the individual transactions
            7. Sort results by TRANSACTIONDATETIME DESC by default unless another sort is specified
            8. Limit to 1000 records max to prevent performance issues
            
            Here is database info: CREATE TABLE [AdyenPaymentTransaction] (
                                                    [PSPREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL, 
                                                    [MERCHANTREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [TRANSACTIONDATETIME] DATETIME NULL, 
                                                    [TIMEZONE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [CURRENCY] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTMETHOD] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTSTATUS] NVARCHAR(20) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [RISKSCORE] INTEGER NULL, 
                                                    CONSTRAINT [PK__AdyenPay__167A345A1489B7D8] PRIMARY KEY ([PSPREFERENCE])
                                                )

                                                /*
                                                3 rows from AdyenPaymentTransaction table:
                                                PSPREFERENCE	MERCHANTREFERENCE	TRANSACTIONDATETIME	TIMEZONE	PAYMENTAMOUNT	CURRENCY	PAYMENTMETHOD	PAYMENTSTATUS	RISKSCORE
                                                B25KD5G9X5STZK65	cfdcccacc90d4a84a20d313c680133fa	2024-04-16 17:57:00	IST	185.94	USD	unknown card	Refused	0
                                                B3J7GSMB4FVJK8V5	c33ad9b319544586b2cb9def832aa6f5	2024-07-08 10:35:00	IST	17.25	USD	Visa	Cancelled	1
                                                B4WPLLG2DTPRCK65	3a97f78e98714365a402f36d12ecc5cb	2024-07-12 11:45:00	IST	66.44	USD	Visa	Cancelled	1
                                                */
                                                CREATE TABLE [BankPaymentTransaction] (
                                                    [STORENUMBER] INTEGER NULL, 
                                                    [CHANNELNAME] VARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [LAST4DIGITS] CHAR(4) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [CARDID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [TRANSACTIONNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [TRANSACTIONDATETIME] DATETIME NULL, 
                                                    [CAPTUREDAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [TRANSACTIONTYPE] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PSPREFERENCE] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL, 
                                                    [PAYMENTMETHOD] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [SETTLEMENTDATE] DATE NULL, 
                                                    [SETTLEMENTID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [MERCHANTID] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [GROSSAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [TRANSACTIONFEES] DECIMAL(10, 2) NULL, 
                                                    [NETAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [CURRENCY] NVARCHAR(10) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [DEPOSITAMOUNT] DECIMAL(10, 2) NULL, 
                                                    [DEPOSITDATE] DATE NULL, 
                                                    [BANKACCOUNTNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [REFERENCENUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [PAYMENTPROVIDER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    [BATCHNUMBER] NVARCHAR(255) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
                                                    CONSTRAINT [PK__BankPaym__2CABD762F33B242E] PRIMARY KEY ([PSPREFERENCE])
                                                )

                                                /*
                                                3 rows from BankPaymentTransaction table:
                                                STORENUMBER	CHANNELNAME	LAST4DIGITS	CARDID	TRANSACTIONNUMBER	TRANSACTIONDATETIME	CAPTUREDAMOUNT	TRANSACTIONTYPE	PSPREFERENCE	PAYMENTMETHOD	SETTLEMENTDATE	SETTLEMENTID	MERCHANTID	GROSSAMOUNT	TRANSACTIONFEES	NETAMOUNT	CURRENCY	DEPOSITAMOUNT	DEPOSITDATE	BANKACCOUNTNUMBER	REFERENCENUMBER	PAYMENTPROVIDER	BATCHNUMBER
                                                5425	Fabrikam call center	1111	VISA	d3271376228543029d0bd99f9a9224a4	2024-07-17 06:44:59	28.82	Authorize	BMXXCC9MFC3GDXT5	CreditCard	2024-10-14	1234	b0183b8f0bec4bd2a47c52b1ecebe8b10	28.82	2.00	26.82	USD	28.82	2024-10-14	2345678901	b0183b8f0bec12	Adyen 	V6ZMHFVSV22WQT22
                                                5413	Fabrikam call center	1111	VISA	090178b06d65458e87e67e66092e63d8	2024-07-17 06:59:49	28.82	Capture	D44GVXL969NKGK82	CreditCard	2024-10-14	1234	b0183b8f0bec4bd2a47c52b1ecebe8b2	28.82	2.00	26.82	USD	28.82	2024-10-14	2345678901	b0183b8f0bec4	Adyen 	V6ZMHFVSV22WQT10
                                                5415	Fabrikam call center	1111	VISA	090178b06d65458e87e67e66092e63d8s	2024-07-17 06:59:49	38.99	Capture	D44GVXL969NKGK84	CreditCard	2024-10-14	1234	b0183b8f0bec4bd2a47c52b1ecebe8b3	38.99	2.00	36.99	USD	38.99	2024-10-14	2345678901	b0183b8f0bec5	Adyen 	V6ZMHFVSV22WQT12
                                                */
            
            Previous chat history context:
            {formatted_chat_history}
            
            Output format: Return only the SQL query with no explanations or additional text.
            """
            
            # Generate the SQL query
            sql_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator."},
                    {"role": "user", "content": query_generation_prompt}
                ],
                temperature=0
            )
            
            # Extract the SQL query
            sql_query = sql_response.choices[0].message.content.strip()
            # Clean up the query (remove backticks, etc.)
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            logger.info(f"Generated SQL query for list request: {sql_query}")
            
            # We'll return the SQL query to the main function, which will then execute it
            # and handle the rest of the processing, including generating the CSV
            
            # Create a simulated result to be processed further
            dummy_result = {
                "sql_query": sql_query,
                "response": "This is a list request that requires CSV generation.",
                "csv_url": None
            }
            
            return json.dumps(dummy_result)
            
        except Exception as e:
            logger.error(f"Error processing list query: {str(e)}", exc_info=True)
            return json.dumps({
                "sql_query": "SELECT TOP 10 * FROM AdyenPaymentTransaction; -- Error generating specific query",
                "response": f"I encountered an error while processing your request for a list of transactions. Please try again with more specific criteria.\n\nError details: {str(e)}",
                "csv_url": None
            })
    
    @kernel_function(
        description="Determine if a CSV file should be generated for a query",
        name="should_generate_csv"
    )
    def should_generate_csv(self, user_input: str, query_result: str) -> str:
        """
        Determine if a CSV file should be generated for a query based on the result size and content.
        
        Args:
            user_input (str): The user's input query
            query_result (str): JSON string containing the query results
            
        Returns:
            str: "true" if CSV should be generated, "false" otherwise
        """
        try:
            # Parse the query result
            result_data = json.loads(query_result)
            
            if not result_data.get("success", False):
                return "false"
            
            row_count = result_data.get("row_count", 0)
            
            # Check if the query mentions list, download, all transactions, etc.
            list_indicators = ["list", "all transactions", "download", "csv", "excel", "export"]
            should_generate = any(indicator in user_input.lower() for indicator in list_indicators)
            
            # Also generate if there are more than 20 rows
            if row_count > 20:
                should_generate = True
                
            return "true" if should_generate else "false"
            
        except Exception as e:
            logger.error(f"Error determining if CSV should be generated: {e}")
            return "false"