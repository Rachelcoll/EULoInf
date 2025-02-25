import os
import logging
from datetime import datetime

log_filename = f"../logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Ensure the logs directory exists
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logger = logging.getLogger("the_logger")
logger.setLevel(logging.DEBUG) 

# File handler (logs everything)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)  # Save all logs

# Console handler (only show INFO and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers (only add if they haven’t been added already)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
