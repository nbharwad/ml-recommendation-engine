"""Structured logging with trace_id propagation."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.config import settings


# Context variable for trace_id propagation across async calls
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def get_trace_id() -> Optional[str]:
    """Get current trace_id from context."""
    return trace_id_var.get()


def set_trace_id(trace_id: Optional[str] = None) -> None:
    """Set trace_id in context. If None, generates a new UUID."""
    trace_id_var.set(trace_id or str(uuid4()))


def clear_trace_id() -> None:
    """Clear trace_id from context."""
    trace_id_var.set(None)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        # Add trace_id if available
        trace_id = trace_id_var.get()
        if trace_id:
            log_data["trace_id"] = trace_id

        # Add duration_ms if present in extra
        if hasattr(record, "duration_ms") and record.duration_ms is not None:
            log_data["duration_ms"] = record.duration_ms

        # Add cache_hit if present in extra
        if hasattr(record, "cache_hit") and record.cache_hit is not None:
            log_data["cache_hit"] = record.cache_hit

        # Add step if present in extra
        if hasattr(record, "step") and record.step:
            log_data["step"] = record.step

        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            for key, value in record.extra.items():
                if key not in log_data:
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        trace_id = trace_id_var.get()
        trace_str = f" [{trace_id[:8]}]" if trace_id else ""
        
        duration = ""
        if hasattr(record, "duration_ms") and record.duration_ms is not None:
            duration = f" ({record.duration_ms}ms)"
        
        step = ""
        if hasattr(record, "step") and record.step:
            step = f" [{record.step}]"
        
        return f"{record.levelname}{trace_str}{step}{duration} {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    """Factory function to get a configured logger.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set level from settings
        logger.setLevel(getattr(logging, settings.log_level))
        
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter based on settings
        if settings.log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(TextFormatter())
        
        logger.addHandler(handler)
        
        # Prevent propagation to root logger (avoids duplicate logs)
        logger.propagate = False
    
    return logger


def log_agent_step(
    logger: logging.Logger,
    step: str,
    message: str,
    duration_ms: Optional[float] = None,
    cache_hit: Optional[bool] = None,
    **extra: Any,
) -> None:
    """Convenience function to log an agent step.
    
    Args:
        logger: Logger instance
        step: Step name (e.g., "planner", "retrieval", "reasoning")
        message: Log message
        duration_ms: Optional duration in milliseconds
        cache_hit: Optional cache hit flag
        **extra: Additional fields to include in log
    """
    # Create extra dict for extra fields
    extra_dict: dict[str, Any] = {"step": step}
    
    if duration_ms is not None:
        extra_dict["duration_ms"] = duration_ms
    
    if cache_hit is not None:
        extra_dict["cache_hit"] = cache_hit
    
    # Add any additional extra fields
    extra_dict.update(extra)
    
    # Use extra parameter to pass custom fields
    logger.info(message, extra=extra_dict)


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    client_ip: Optional[str] = None,
) -> None:
    """Log HTTP request.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        client_ip: Client IP address
    """
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }
    
    if client_ip:
        log_data["client_ip"] = client_ip
    
    level = logging.INFO if status_code < 400 else logging.WARNING
    logger.log(level, f"{method} {path} {status_code}", extra=log_data)


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[dict[str, Any]] = None,
) -> None:
    """Log error with context.
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context dict
    """
    extra = {"exception_type": type(error).__name__}
    
    if context:
        extra["context"] = context
    
    logger.error(str(error), extra=extra, exc_info=True)