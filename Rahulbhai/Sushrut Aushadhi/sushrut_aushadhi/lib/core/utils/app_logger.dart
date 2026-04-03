import 'package:flutter/foundation.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';

enum LogLevel {
  debug,
  info,
  warning,
  error,
}

class AppLogger {
  static bool _enableLogging = true;
  static bool _sendToCrashlytics = true;

  static void init({
    bool enableLogging = true,
    bool sendToCrashlytics = true,
  }) {
    _enableLogging = enableLogging;
    _sendToCrashlytics = sendToCrashlytics;
  }

  static void debug(String message, {String? tag}) {
    _log(LogLevel.debug, message, tag: tag);
  }

  static void info(String message, {String? tag}) {
    _log(LogLevel.info, message, tag: tag);
  }

  static void warning(String message, {String? tag}) {
    _log(LogLevel.warning, message, tag: tag);
  }

  static void error(String message, {dynamic error, StackTrace? stackTrace, String? tag}) {
    _log(LogLevel.error, message, tag: tag);
    
    if (_sendToCrashlytics && error != null) {
      FirebaseCrashlytics.instance.recordError(error, stackTrace);
    }
  }

  static void _log(LogLevel level, String message, {String? tag}) {
    if (!_enableLogging && !kDebugMode) return;

    final timestamp = DateTime.now().toIso8601String();
    final levelStr = level.name.toUpperCase().padRight(7);
    final tagStr = tag != null ? '[$tag] ' : '';
    
    final logMessage = '[$timestamp] $levelStr $tagStr$message';
    
    if (kDebugMode) {
      switch (level) {
        case LogLevel.debug:
        case LogLevel.info:
          debugPrint(logMessage);
        case LogLevel.warning:
        case LogLevel.error:
          debugPrint(logMessage);
      }
    }
  }

  static void setUserIdentifier(String userId) {
    if (_sendToCrashlytics) {
      FirebaseCrashlytics.instance.setUserIdentifier(userId);
    }
  }

  static void logCustomKey(String key, String value) {
    if (_sendToCrashlytics) {
      FirebaseCrashlytics.instance.setCustomKey(key, value);
    }
  }
}
