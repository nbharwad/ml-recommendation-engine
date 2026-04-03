import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../error/exceptions.dart';
import '../error/result.dart';

class ResultHandler {
  static Widget handle<T>({
    required Result<T> result,
    required Widget Function(T data) success,
    required Widget Function(String message) loading,
    required Widget Function(String message, String? code) error,
  }) {
    return switch (result) {
      Success<T>(:final data) => success(data),
      Failure<T>(:final message, :final code) => error(message, code),
    };
  }

  static Widget when<T>({
    required AsyncValue<T> asyncValue,
    required Widget Function(T data) data,
    Widget Function()? loading,
    Widget Function(Object error, StackTrace? stack)? error,
  }) {
    return asyncValue.when(
      data: data,
      loading: () => loading?.call() ?? const Center(child: CircularProgressIndicator()),
      error: (e, st) => error?.call(e, st) ?? _defaultError(e),
    );
  }

  static Widget _defaultError(Object error) {
    final message = error is AppException ? error.message : 'Something went wrong';
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.error_outline, size: 48, color: Colors.red),
          const SizedBox(height: 16),
          Text(message, textAlign: TextAlign.center),
        ],
      ),
    );
  }

  static String getErrorMessage(Object error) {
    if (error is AppException) return error.message;
    if (error is FirebaseException) return error.message ?? 'Database error';
    if (error is FormatException) return 'Invalid input format';
    return 'An unexpected error occurred';
  }

  static ScaffoldFeatureController<SnackBar, SnackBarClosedReason> showErrorSnackbar(
    BuildContext context,
    Object error, {
    Duration duration = const Duration(seconds: 4),
  }) {
    return ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(getErrorMessage(error)),
        backgroundColor: Colors.red,
        duration: duration,
      ),
    );
  }

  static ScaffoldFeatureController<SnackBar, SnackBarClosedReason> showSuccessSnackbar(
    BuildContext context,
    String message, {
    Duration duration = const Duration(seconds: 3),
  }) {
    return ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
        duration: duration,
      ),
    );
  }
}
