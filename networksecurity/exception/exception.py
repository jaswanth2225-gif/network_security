"""
Custom error class for the Network Security project.

Purpose:
- Show WHAT error happened (error message)
- Show WHERE it happened (file name and line number)
- Make debugging much faster (know exactly where to look)
"""

import sys  # System module (used to get error details like file name and line number)
from networksecurity.logging.logger import logging as _logging

# Create a logger for this file using central configuration
logger = _logging.getLogger(__name__)


class NetworkSecurityException(Exception):
    """
    Custom exception that shows error message + file name + line number.
    
    Why custom exception?
    - Default errors don't show file name
    - Default errors don't show line number  
    - We added both to help debugging faster
    """

    def __init__(self, error_message, error_details: sys):
        """
        Initialize the custom error.
        
        error_message = Original Python error (e.g., "Division by zero")
        error_details = sys module (contains traceback = where error happened)
        """

        self.error_message = error_message  # Store the error message

        # Get traceback (detailed record of error location)
        # exc_info() returns (error_type, error_value, traceback) ← we only need traceback
        _, _, exc_tb = error_details.exc_info()

        # Extract line number from traceback
        self.lineno = exc_tb.tb_lineno  # tb_lineno = Line number where error occurred

        # Extract file name from traceback
        self.file_name = exc_tb.tb_frame.f_code.co_filename  # File where error happened

    def __str__(self):
        """
        Format the error message for display.
        
        This controls HOW the error looks when printed.
        Example output: "Error occurred in python script name [main.py] line number [42] error message [Division by zero]"
        """
        return (
            f"Error occurred in python script name [{self.file_name}] "
            f"line number [{self.lineno}] "
            f"error message [{self.error_message}]"
        )


# Run this block only when this file is executed directly (not when imported)
if __name__ == "__main__":

    try:
        logger.info("Inside try block")

        # This line will cause an error (division by zero) to test the custom exception
        a = 1 / 0

    except Exception as e:
        # Catch the error and raise our custom exception (shows file, line, and error message)
        raise NetworkSecurityException(e, sys)













