"""
Logging Configuration Module.

This module sets up centralized logging for the entire project.
All log messages are written to timestamped files in the 'logs' directory.

Logging helps in:
1. Debugging issues by tracking execution flow
2. Monitoring pipeline progress
3. Recording errors and warnings
4. Creating audit trails for production systems
"""

import logging
import os
from datetime import datetime

# Create 'logs' directory in current working directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Generate timestamped log filename
# Format: log_YYYY-MM-DD_HH-MM-SS.log
# Example: log_2025-12-23_14-30-45.log
log_file = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_file_path = os.path.join(logs_path, log_file)

# Configure logging with:
# - filename: Where to write logs
# - level: Minimum severity to log (INFO captures INFO, WARNING, ERROR, CRITICAL)
# - format: How each log entry is structured
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,  # Log INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Timestamp - Level - Message
)

# Usage in other modules:
# from networksecurity.logging.logger import logging
# logging.info("This is an info message")
# logging.warning("This is a warning")
# logging.error("This is an error")
