import 'package:firebase_performance/firebase_performance.dart';

class PerformanceService {
  static bool _isEnabled = true;

  static void init({bool isEnabled = true}) {
    _isEnabled = isEnabled;
  }

  static Trace startTrace(String name) {
    if (!_isEnabled) {
      return FirebasePerformance.instance.newTrace(name);
    }
    final trace = FirebasePerformance.instance.newTrace(name);
    trace.start();
    return trace;
  }

  static Future<T> traceFuture<T>(
    String name,
    Future<T> Function() operation, {
    Map<String, String>? attributes,
  }) async {
    final trace = startTrace(name);
    
    if (attributes != null) {
      for (final entry in attributes.entries) {
        trace.putAttribute(entry.key, entry.value);
      }
    }

    try {
      final result = await operation();
      return result;
    } catch (e) {
      trace.putAttribute('error', e.toString());
      rethrow;
    } finally {
      trace.stop();
    }
  }

  static Future<T> traceFutureWithMetric<T>(
    String name,
    Future<T> Function() operation, {
    String? metricName,
    String? metricValue,
    Map<String, String>? attributes,
  }) async {
    final trace = startTrace(name);
    
    if (attributes != null) {
      for (final entry in attributes.entries) {
        trace.putAttribute(entry.key, entry.value);
      }
    }

    try {
      final result = await operation();
      
      if (metricName != null && metricValue != null) {
        trace.putAttribute(metricName, metricValue);
      }
      
      return result;
    } catch (e) {
      trace.putAttribute('error', e.toString());
      rethrow;
    } finally {
      trace.stop();
    }
  }
}