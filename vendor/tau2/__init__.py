import os
import sys

from loguru import logger

from vendor.tau2.config import DEFAULT_LOG_LEVEL

# Remove default handler to avoid duplicate logs
logger.remove()

# Get log level from environment variable, then tau2 config, then default to WARNING
log_level = os.environ.get("TAU2_LOG_LEVEL")
if log_level is None:
    log_level = DEFAULT_LOG_LEVEL

# Add handler with appropriate log level
logger.add(
    sys.stderr,
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
)
