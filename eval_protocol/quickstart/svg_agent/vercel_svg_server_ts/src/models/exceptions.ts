/**
 * OpenAI error to Status code mapping for proper error handling
 *
 * Port of the Python exception handling from eval_protocol/exceptions.py
 * Maps OpenAI SDK errors to appropriate Status codes for consistent error handling
 */

import OpenAI from 'openai';
import { Status, StatusCode } from './status.js';

// Base class for eval protocol errors
export class EvalProtocolError extends Error {
  statusCode: StatusCode;

  constructor(message: string, statusCode: StatusCode) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
  }
}

// Specific exception classes
export class CancelledError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.CANCELLED);
  }
}

export class UnknownError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.UNKNOWN);
  }
}

export class InvalidArgumentError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.INVALID_ARGUMENT);
  }
}

export class DeadlineExceededError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.DEADLINE_EXCEEDED);
  }
}

export class NotFoundError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.NOT_FOUND);
  }
}

export class AlreadyExistsError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.ALREADY_EXISTS);
  }
}

export class PermissionDeniedError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.PERMISSION_DENIED);
  }
}

export class ResourceExhaustedError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.RESOURCE_EXHAUSTED);
  }
}

export class FailedPreconditionError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.FAILED_PRECONDITION);
  }
}

export class AbortedError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.ABORTED);
  }
}

export class OutOfRangeError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.OUT_OF_RANGE);
  }
}

export class UnimplementedError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.UNIMPLEMENTED);
  }
}

export class InternalError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.INTERNAL);
  }
}

export class UnavailableError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.UNAVAILABLE);
  }
}

export class DataLossError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.DATA_LOSS);
  }
}

export class UnauthenticatedError extends EvalProtocolError {
  constructor(message: string = '') {
    super(message, StatusCode.UNAUTHENTICATED);
  }
}

// Mapping from status codes to exception classes
const STATUS_CODE_TO_EXCEPTION = new Map<StatusCode, typeof EvalProtocolError | null>([
  [StatusCode.OK, null],
  [StatusCode.CANCELLED, CancelledError],
  [StatusCode.UNKNOWN, UnknownError],
  [StatusCode.INVALID_ARGUMENT, InvalidArgumentError],
  [StatusCode.DEADLINE_EXCEEDED, DeadlineExceededError],
  [StatusCode.NOT_FOUND, NotFoundError],
  [StatusCode.ALREADY_EXISTS, AlreadyExistsError],
  [StatusCode.PERMISSION_DENIED, PermissionDeniedError],
  [StatusCode.RESOURCE_EXHAUSTED, ResourceExhaustedError],
  [StatusCode.FAILED_PRECONDITION, FailedPreconditionError],
  [StatusCode.ABORTED, AbortedError],
  [StatusCode.OUT_OF_RANGE, OutOfRangeError],
  [StatusCode.UNIMPLEMENTED, UnimplementedError],
  [StatusCode.INTERNAL, InternalError],
  [StatusCode.UNAVAILABLE, UnavailableError],
  [StatusCode.DATA_LOSS, DataLossError],
  [StatusCode.UNAUTHENTICATED, UnauthenticatedError],
  [StatusCode.FINISHED, null],  // Success, no exception
  [StatusCode.RUNNING, null],   // In progress, no exception
  [StatusCode.SCORE_INVALID, null] // Success, no exception
]);

/**
 * Create an exception instance for a given status code
 */
export function exceptionForStatusCode(code: StatusCode, message: string = ''): EvalProtocolError | null {
  const exceptionClass = STATUS_CODE_TO_EXCEPTION.get(code);
  if (!exceptionClass) {
    return null;
  }
  return new exceptionClass(message, code);
}

/**
 * Map OpenAI errors to appropriate Status objects
 * This is the main function used in the rollout processor
 */
export function mapOpenAIErrorToStatus(error: any): Status {
  const errorMessage = error.message || String(error);

  // Check if it's an OpenAI error
  if (error instanceof OpenAI.AuthenticationError) {
    return Status.rolloutPermissionDeniedError(errorMessage);
  }

  if (error instanceof OpenAI.PermissionDeniedError) {
    return Status.rolloutPermissionDeniedError(errorMessage);
  }

  if (error instanceof OpenAI.NotFoundError) {
    return Status.rolloutNotFoundError(errorMessage);
  }

  if (error instanceof OpenAI.RateLimitError) {
    return Status.rolloutResourceExhaustedError(errorMessage);
  }

  if (error instanceof OpenAI.BadRequestError) {
    return Status.rolloutInvalidArgumentError(errorMessage);
  }

  if (error instanceof OpenAI.InternalServerError) {
    return Status.rolloutInternalError(errorMessage);
  }

  if (error instanceof OpenAI.UnprocessableEntityError) {
    return Status.rolloutInvalidArgumentError(errorMessage);
  }

  if (error instanceof OpenAI.APIConnectionTimeoutError) {
    return Status.rolloutDeadlineExceededError(errorMessage);
  }

  if (error instanceof OpenAI.ConflictError) {
    return Status.rolloutAlreadyExistsError(errorMessage);
  }

  // Check for network/connection errors
  if (error.code === 'ECONNRESET' || error.code === 'ENOTFOUND' || error.code === 'ETIMEDOUT') {
    return Status.rolloutUnavailableError(errorMessage);
  }

  // Default to internal error for unknown errors
  return Status.rolloutInternalError(errorMessage);
}

/**
 * Check if an error should be retried based on its type
 * Mirrors the retry logic from Python eval_protocol/pytest/exception_config.py
 */
export function isRetryableError(error: any): boolean {
  // OpenAI errors that should be retried
  if (error instanceof OpenAI.RateLimitError ||
      error instanceof OpenAI.InternalServerError ||
      error instanceof OpenAI.APIConnectionTimeoutError ||
      error instanceof OpenAI.APIConnectionError) {
    return true;
  }

  // Network/connection errors
  if (error.code === 'ECONNRESET' ||
      error.code === 'ENOTFOUND' ||
      error.code === 'ETIMEDOUT' ||
      error.code === 'ECONNREFUSED') {
    return true;
  }

  // Eval Protocol errors that should be retried
  if (error instanceof UnknownError ||
      error instanceof DeadlineExceededError ||
      error instanceof NotFoundError ||
      error instanceof PermissionDeniedError ||
      error instanceof UnavailableError ||
      error instanceof UnauthenticatedError ||
      error instanceof ResourceExhaustedError) {
    return true;
  }

  return false;
}
