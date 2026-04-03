import 'package:firebase_remote_config/firebase_remote_config.dart';
import 'package:flutter/foundation.dart';

class RemoteConfigService {
  static final FirebaseRemoteConfig _config = FirebaseRemoteConfig.instance;
  static bool _initialized = false;

  static const Map<String, dynamic> _defaults = {
    'store_phone': '',
    'store_whatsapp': '',
    'store_email': '',
    'store_address': '',
    'store_name': 'Sushrut Aushadhi',
    'drug_license_no': '',
    'gst_number': '',
    'news_api_key': '',
    'algolia_app_id': '',
    'algolia_api_key': '',
    'delivery_hours': '2',
    'min_order_amount': 0.0,
    'free_delivery_above': 200.0,
    'first_order_discount': 20,
    'promo_code': 'SUSHRUT20',
    'store_open_time': '09:00',
    'store_close_time': '21:00',
    'is_store_open': true,
    'maintenance_mode': false,
    'maintenance_message': 'App is under maintenance. We will be back soon!',
  };

  static Future<void> initialize() async {
    if (_initialized) return;

    try {
      await _config.setConfigSettings(
        RemoteConfigSettings(
          fetchTimeout: const Duration(seconds: 15),
          minimumFetchInterval: kDebugMode
              ? const Duration(minutes: 1)
              : const Duration(hours: 1),
        ),
      );

      await _config.setDefaults(_defaults);
      await _config.fetchAndActivate();

      _initialized = true;
      debugPrint('Remote Config initialized successfully');
    } catch (e) {
      debugPrint('Remote Config fetch failed: $e. Using default values.');
      _initialized = true;
    }
  }

  static Future<void> refresh() async {
    try {
      await _config.fetchAndActivate();
    } catch (e) {
      debugPrint('Remote Config refresh failed: $e');
    }
  }

  static String get storePhone => _config.getString('store_phone');
  static String get storeWhatsApp => _config.getString('store_whatsapp');
  static String get storeEmail => _config.getString('store_email');
  static String get storeAddress => _config.getString('store_address');
  static String get storeName => _config.getString('store_name');
  static String get drugLicenseNo => _config.getString('drug_license_no');
  static String get gstNumber => _config.getString('gst_number');
  static String get newsApiKey => _config.getString('news_api_key');
  static String get algoliaAppId => _config.getString('algolia_app_id');
  static String get algoliaApiKey => _config.getString('algolia_api_key');
  static String get deliveryHours => _config.getString('delivery_hours');
  static double get minOrderAmount => _config.getDouble('min_order_amount');
  static double get freeDeliveryAbove => _config.getDouble('free_delivery_above');
  static int get firstOrderDiscount => _config.getInt('first_order_discount');
  static String get promoCode => _config.getString('promo_code');
  static String get storeOpenTime => _config.getString('store_open_time');
  static String get storeCloseTime => _config.getString('store_close_time');
  static bool get isStoreOpen => _config.getBool('is_store_open');
  static bool get maintenanceMode => _config.getBool('maintenance_mode');
  static String get maintenanceMessage => _config.getString('maintenance_message');

  static bool get isCurrentlyOpen {
    try {
      if (!isStoreOpen) return false;

      final now = DateTime.now();
      final openParts = storeOpenTime.split(':');
      final closeParts = storeCloseTime.split(':');

      final openTime = DateTime(
        now.year, now.month, now.day,
        int.parse(openParts[0]),
        int.parse(openParts[1]),
      );
      final closeTime = DateTime(
        now.year, now.month, now.day,
        int.parse(closeParts[0]),
        int.parse(closeParts[1]),
      );

      return now.isAfter(openTime) && now.isBefore(closeTime);
    } catch (e) {
      return true;
    }
  }
}