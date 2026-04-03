import 'package:firebase_analytics/firebase_analytics.dart';

class AnalyticsService {
  static bool _isEnabled = true;

  static void init({bool isEnabled = true}) {
    _isEnabled = isEnabled;
  }

  static Future<void> trackEvent({
    required String name,
    Map<String, dynamic>? parameters,
  }) async {
    if (!_isEnabled) {
      return;
    }

    final params = parameters?.map(
      (key, value) => MapEntry(key, value is num ? value : value.toString()),
    );

    await FirebaseAnalytics.instance.logEvent(
      name: name,
      parameters: params,
    );
  }

  static Future<void> trackMedicineView({
    required String medicineId,
    required String medicineName,
  }) async {
    await trackEvent(
      name: 'medicine_view',
      parameters: {
        'medicine_id': medicineId,
        'medicine_name': medicineName,
      },
    );
  }

  static Future<void> trackAddToCart({
    required String medicineId,
    required String medicineName,
    required double price,
    required int quantity,
  }) async {
    await trackEvent(
      name: 'add_to_cart',
      parameters: {
        'medicine_id': medicineId,
        'medicine_name': medicineName,
        'price': price,
        'quantity': quantity,
      },
    );
  }

  static Future<void> trackOrderPlaced({
    required String orderId,
    required double totalAmount,
    required int itemCount,
  }) async {
    await trackEvent(
      name: 'order_placed',
      parameters: {
        'order_id': orderId,
        'total_amount': totalAmount,
        'item_count': itemCount,
      },
    );
  }

  static Future<void> trackPrescriptionUploaded({
    required String prescriptionId,
  }) async {
    await trackEvent(
      name: 'prescription_uploaded',
      parameters: {
        'prescription_id': prescriptionId,
      },
    );
  }

  static Future<void> trackLoginSuccess({
    required String userId,
  }) async {
    await trackEvent(
      name: 'login_success',
      parameters: {
        'user_id': userId,
      },
    );
  }

  static Future<void> trackLoginFail({
    required String errorCode,
  }) async {
    await trackEvent(
      name: 'login_fail',
      parameters: {
        'error_code': errorCode,
      },
    );
  }
}