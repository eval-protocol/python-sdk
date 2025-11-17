import OpenAI from 'openai';
import { Status, StatusCode } from './status.js';

export class EvalProtocolError extends Error {
  statusCode: StatusCode;

  constructor(message: string, statusCode: StatusCode) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
  }
}

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

type EvalProtocolErrorConstructor = new (message?: string) => EvalProtocolError;

const STATUS_CODE_TO_EXCEPTION = new Map<StatusCode, EvalProtocolErrorConstructor | null>([
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
  [StatusCode.FINISHED, null],
  [StatusCode.RUNNING, null],
  [StatusCode.SCORE_INVALID, null],
]);

export function exceptionForStatusCode(code: StatusCode, message: string = ''): EvalProtocolError | null {
  const exceptionClass = STATUS_CODE_TO_EXCEPTION.get(code);
  if (!exceptionClass) {
    return null;
  }
  return new exceptionClass(message);
}

export function mapOpenAIErrorToStatus(error: any): Status {
  const errorMessage = error.message || String(error);

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

  if (error.code === 'ECONNRESET' || error.code === 'ENOTFOUND' || error.code === 'ETIMEDOUT') {
    return Status.rolloutUnavailableError(errorMessage);
  }

  return Status.rolloutInternalError(errorMessage);
}

export function isRetryableError(error: any): boolean {
  if (error instanceof OpenAI.RateLimitError ||
      error instanceof OpenAI.InternalServerError ||
      error instanceof OpenAI.APIConnectionTimeoutError ||
      error instanceof OpenAI.APIConnectionError) {
    return true;
  }

  if (error.code === 'ECONNRESET' ||
      error.code === 'ENOTFOUND' ||
      error.code === 'ETIMEDOUT' ||
      error.code === 'ECONNREFUSED') {
    return true;
  }

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
