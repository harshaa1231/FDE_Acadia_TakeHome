"""Structured application errors mapped to HTTP responses.

Every error the API returns to a caller goes through AppError so the
response body is always {"error": {"code", "message", "details"?}} and never
a raw stack trace.
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.details:
            body["error"]["details"] = self.details
        return body


class ValidationError(AppError):
    status_code = 400
    code = "validation_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class UnanswerableError(AppError):
    """Raised when a question cannot be answered from the dataset. Refusing
    is the correct behavior, but it is still communicated as a normal,
    structured API response rather than a failure state."""

    status_code = 422
    code = "unanswerable"


class QueueFullError(AppError):
    status_code = 429
    code = "queue_full"


class InternalError(AppError):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str = "An internal error occurred.", *, details=None):
        # Never leak the original exception message/stack trace to the caller.
        super().__init__(message, details=details)
