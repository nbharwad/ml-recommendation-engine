abstract class AppException implements Exception {
  final String message;
  final String? code;
  final dynamic originalError;

  AppException({
    required this.message,
    this.code,
    this.originalError,
  });

  @override
  String toString() => message;
}

class AuthException extends AppException {
  AuthException({
    required super.message,
    super.code,
    super.originalError,
  });
}

class FirestoreException extends AppException {
  FirestoreException({
    required super.message,
    super.code,
    super.originalError,
  });
}

class ValidationException extends AppException {
  final Map<String, String>? fieldErrors;

  ValidationException({
    required super.message,
    this.fieldErrors,
    super.code,
  });
}

class NetworkException extends AppException {
  NetworkException({
    required super.message,
    super.code,
    super.originalError,
  });
}

class RateLimitException extends AppException {
  final DateTime? retryAfter;

  RateLimitException({
    required super.message,
    this.retryAfter,
    super.code,
  });
}

class StorageException extends AppException {
  StorageException({
    required super.message,
    super.code,
    super.originalError,
  });
}

class PermissionException extends AppException {
  PermissionException({
    required super.message,
    super.code,
  });
}
