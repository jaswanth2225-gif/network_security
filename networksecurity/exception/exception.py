"""
Custom Exception Module for Network Security Project.

This module defines a custom exception class that provides enhanced
error reporting with file names, line numbers, and detailed error messages.
This helps in debugging by pinpointing exactly where errors occur.
"""

import sys
import logging
from venv import logger

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class NetworkSecurityException(Exception):
    """
    Custom exception class for the Network Security ML project.
    
    This exception extends Python's base Exception class to provide
    detailed error information including:
    - Original error message
    - File name where error occurred
    - Line number where error occurred
    
    Attributes:
        error_message: The original exception message
        lineno (int): Line number where exception was raised
        file_name (str): Python script name where exception occurred
    """
    
    def __init__(self, error_message, error_details: sys):
        """
        Initialize custom exception with enhanced error details.
        
        Args:
            error_message: Original exception object or error string
            error_details (sys): sys module to extract traceback information
        """
        self.error_message = error_message

        # Extract exception traceback information
        # exc_info() returns (type, value, traceback)
        _, _, exc_tb = error_details.exc_info()

        # Get line number where exception occurred
        self.lineno = exc_tb.tb_lineno
        
        # Get the filename from the code object
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        """
        Return formatted error message string.
        
        Returns:
            str: Formatted error message with file name, line number, and error details
            
        Example:
            Error occurred in python script name [/path/to/file.py] 
            line number [42] error message[division by zero]
        """
        return "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )


# Test/demonstration block
if __name__ == "__main__":
    """
    Test the custom exception class.
    
    This block demonstrates how NetworkSecurityException works
    by intentionally causing a division by zero error.
    """
    try:
        logger.info("Enter the try block")
        a = 1/0  # This will raise ZeroDivisionError
        print("This will not be printed", a)
    except Exception as e:
        # Wrap the base exception in our custom exception
        raise NetworkSecurityException(e, sys)