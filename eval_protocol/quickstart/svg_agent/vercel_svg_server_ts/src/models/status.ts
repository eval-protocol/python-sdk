/**
 * AIP-193 compatible Status model for standardized error responses.
 *
 * This model follows Google's AIP-193 standard for error handling:
 * https://google.aip.dev/193
 *
 * Port of the Python Status class from eval_protocol/models.py
 */

export enum StatusCode {
  // Standard gRPC codes
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
  UNAUTHENTICATED = 16,

  // Custom codes for EP (using higher numbers to avoid conflicts)
  FINISHED = 100,
  RUNNING = 101,
  SCORE_INVALID = 102
}

export interface ErrorInfo {
  '@type': string;
  reason?: string;
  domain?: string;
  metadata?: Record<string, any>;
}

export interface StatusDetails {
  code: StatusCode;
  message: string;
  details: ErrorInfo[];
}

export class Status {
  code: StatusCode;
  message: string;
  details: ErrorInfo[];

  constructor(code: StatusCode, message: string, details: ErrorInfo[] = []) {
    this.code = code;
    this.message = message;
    this.details = details;
  }

  // Helper method to build details with extra info
  private static _buildDetailsWithExtraInfo(extraInfo?: Record<string, any>): ErrorInfo[] {
    if (!extraInfo) {
      return [];
    }
    return [{
      '@type': 'type.googleapis.com/google.rpc.ErrorInfo',
      reason: 'EXTRA_INFO',
      domain: 'eval-protocol.com',
      metadata: extraInfo
    }];
  }

  // Core factory methods
  static ok(): Status {
    return new Status(StatusCode.OK, 'Success');
  }

  static rolloutRunning(): Status {
    return new Status(StatusCode.RUNNING, 'Rollout is running');
  }

  static evalRunning(): Status {
    return new Status(StatusCode.RUNNING, 'Evaluation is running');
  }

  static rolloutFinished(): Status {
    return new Status(StatusCode.FINISHED, 'Rollout finished');
  }

  static evalFinished(): Status {
    return new Status(StatusCode.FINISHED, 'Evaluation finished');
  }

  // CANCELLED = 1
  static rolloutCancelledError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.cancelledError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static cancelledError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.CANCELLED, errorMessage, details);
  }

  // UNKNOWN = 2
  static rolloutUnknownError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.unknownError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static unknownError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.UNKNOWN, errorMessage, details);
  }

  // INVALID_ARGUMENT = 3
  static rolloutInvalidArgumentError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.invalidArgumentError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static invalidArgumentError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.INVALID_ARGUMENT, errorMessage, details);
  }

  // DEADLINE_EXCEEDED = 4
  static rolloutDeadlineExceededError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.deadlineExceededError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static deadlineExceededError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.DEADLINE_EXCEEDED, errorMessage, details);
  }

  // NOT_FOUND = 5
  static rolloutNotFoundError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.notFoundError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static notFoundError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.NOT_FOUND, errorMessage, details);
  }

  // ALREADY_EXISTS = 6
  static rolloutAlreadyExistsError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.alreadyExistsError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static alreadyExistsError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.ALREADY_EXISTS, errorMessage, details);
  }

  // PERMISSION_DENIED = 7
  static rolloutPermissionDeniedError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.permissionDeniedError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static permissionDeniedError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.PERMISSION_DENIED, errorMessage, details);
  }

  // RESOURCE_EXHAUSTED = 8
  static rolloutResourceExhaustedError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.resourceExhaustedError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static resourceExhaustedError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.RESOURCE_EXHAUSTED, errorMessage, details);
  }

  // FAILED_PRECONDITION = 9
  static rolloutFailedPreconditionError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.failedPreconditionError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static failedPreconditionError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.FAILED_PRECONDITION, errorMessage, details);
  }

  // ABORTED = 10
  static rolloutAbortedError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.abortedError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static abortedError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.ABORTED, errorMessage, details);
  }

  // OUT_OF_RANGE = 11
  static rolloutOutOfRangeError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.outOfRangeError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static outOfRangeError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.OUT_OF_RANGE, errorMessage, details);
  }

  // UNIMPLEMENTED = 12
  static rolloutUnimplementedError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.unimplementedError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static unimplementedError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.UNIMPLEMENTED, errorMessage, details);
  }

  // INTERNAL = 13
  static rolloutInternalError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.internalError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static internalError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.INTERNAL, errorMessage, details);
  }

  // For backwards compatibility
  static rolloutError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.internalError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static error(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.INTERNAL, errorMessage, details);
  }

  // UNAVAILABLE = 14
  static rolloutUnavailableError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.unavailableError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static unavailableError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.UNAVAILABLE, errorMessage, details);
  }

  // DATA_LOSS = 15
  static rolloutDataLossError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.dataLossError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static dataLossError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.DATA_LOSS, errorMessage, details);
  }

  // UNAUTHENTICATED = 16
  static rolloutUnauthenticatedError(errorMessage: string, extraInfo?: Record<string, any>): Status {
    return Status.unauthenticatedError(errorMessage, Status._buildDetailsWithExtraInfo(extraInfo));
  }

  static unauthenticatedError(errorMessage: string, details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.UNAUTHENTICATED, errorMessage, details);
  }

  // Special cases
  static scoreInvalid(message: string = 'Score is invalid', details: ErrorInfo[] = []): Status {
    return new Status(StatusCode.SCORE_INVALID, message, details);
  }

  // Serialization
  toJSON(): StatusDetails {
    return {
      code: this.code,
      message: this.message,
      details: this.details
    };
  }

  static fromJSON(json: StatusDetails): Status {
    return new Status(json.code, json.message, json.details);
  }
}
