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

import logging  # Python's built-in module for recording events/messages during code execution
import os  # Needed to create folders and get current directory
# datetime import removed - not used in this file

# Create 'logs' directory in current working directory if it doesn't exist 
# os.getcwd() → Gets the current project folder (wherever main.py is running from)
# "logs" → Folder name where all log files will be stored

logs_path = os.path.join(os.getcwd(), "logs")  # Combines path like "/project" + "logs" = "/project/logs"
os.makedirs(logs_path, exist_ok=True)  # Create folder if missing; if it exists, don't error out


# Use a single append-only log file to aggregate all pipeline runs into one file
log_file_path = os.path.join(logs_path, "pipeline.log")  # Final path: logs/pipeline.log


logger = logging.getLogger()  # Root logger: The main logger that captures ALL messages throughout the app
logger.setLevel(logging.INFO)  # Only show INFO level and above (skips DEBUG, shows INFO, WARNING, ERROR, CRITICAL)

# Avoid adding duplicate handlers if this module is imported multiple times
# This check prevents creating 5 FileHandlers if logger.py is imported 5 times
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file_path) for h in logger.handlers):
    # FileHandler: Sends log messages to a file on disk (like writing to a diary)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # 'a' = append mode (adds to existing file)
    file_handler.setLevel(logging.INFO)  # FileHandler also respects INFO level minimum
    # Formatter: Defines HOW each log message looks (adds timestamp, level, and message text)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Example output: "2025-12-25 10:30:45 - INFO - Data loaded successfully"
    file_handler.setFormatter(formatter)  # Apply the format to this handler
    logger.addHandler(file_handler)  # Attach the file handler to the main logger

# Also log to console for real-time feedback (watch logs while code runs)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    # StreamHandler: Sends log messages to the console screen (like printing to terminal)
    stream_handler = logging.StreamHandler()  # StreamHandler writes to console/terminal (stdout)
    stream_handler.setLevel(logging.INFO)  # Only show INFO level and above on console
    # Formatter: Same format as file logs so console and file look consistent
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)  # Apply the format to this handler
    logger.addHandler(stream_handler)  # Attach the console handler to the main logger

# Usage in other modules:
# from networksecurity.logging.logger import logging
# logging.info("This is an info message") 
# logging.warning("This is a warning")
# logging.error("This is an error")
