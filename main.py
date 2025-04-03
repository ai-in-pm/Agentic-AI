"""
Agentic AI Demo - Main Entry Point
This script initializes and runs the Agentic AI application.
"""
import os
import sys
import logging
from dotenv import load_dotenv
from api.app import start_api_server
from frontend.app import start_frontend

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    logger.info("Starting Agentic AI Demo Application")

    # Start the API server in a separate process
    import multiprocessing
    api_process = multiprocessing.Process(target=start_api_server)
    api_process.start()

    # Start the frontend
    start_frontend()

    # Clean up when the frontend exits
    api_process.terminate()
    api_process.join()

if __name__ == "__main__":
    main()
