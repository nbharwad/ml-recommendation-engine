import 'dart:convert';
import 'package:flutter/services.dart';

class LocalDataService {
  static Map<String, dynamic>? _cache;
  
  static Map<String, dynamic> normalizeMedicine(Map<String, dynamic> medicine) {
    double parseDouble(dynamic value) {
      if (value is num) return value.toDouble();
      return double.tryParse(value?.toString() ?? '') ?? 0;
    }

    int parseInt(dynamic value, {int fallback = 0}) {
      if (value is int) return value;
      if (value is num) return value.toInt();
      return int.tryParse(value?.toString() ?? '') ?? fallback;
    }

    return {
      'id': medicine['id']?.toString() ?? '',
      'name': medicine['name']?.toString() ?? '',
      'genericName': medicine['composition']?.toString() ??
          medicine['genericName']?.toString() ??
          '',
      'composition': medicine['composition']?.toString() ??
          medicine['genericName']?.toString() ??
          '',
      'manufacturer': medicine['manufacturer']?.toString() ??
          medicine['brand']?.toString() ??
          '',
      'brand': medicine['brand']?.toString() ??
          medicine['manufacturer']?.toString() ??
          '',
      'category': medicine['category']?.toString() ?? 'other',
      'price': parseDouble(medicine['price']),
      'mrp': parseDouble(medicine['mrp'] ?? medicine['price']),
      'stock': parseInt(medicine['stock'], fallback: 50),
      'unit': medicine['unit']?.toString() ?? 'strip',
      'imageUrl': medicine['imageUrl']?.toString() ?? '',
      'requiresPrescription': medicine['requiresPrescription'] == true,
      'description': medicine['composition']?.toString() ??
          medicine['description']?.toString() ??
          '',
      'isActive': medicine['isActive'] is bool ? medicine['isActive'] : true,
    };
  }

  static Future<Map<String, dynamic>> loadData() async {
    if (_cache != null) return _cache!;
    final String jsonString = await rootBundle.loadString(
      'assets/data/medicines.json'
    );
    _cache = json.decode(jsonString);
    return _cache!;
  }

  static Future<List<dynamic>> getMedicines({String? category}) async {
    final data = await loadData();
    final medicines = data['medicines'] as List;
    final normalized =
        medicines.map((m) => normalizeMedicine(Map<String, dynamic>.from(m))).toList();
    if (category == null || category == 'all') return normalized;
    return normalized.where((m) => m['category'] == category).toList();
  }

  static Future<List<dynamic>> searchMedicines(String query) async {
    final data = await loadData();
    final medicines = data['medicines'] as List;
    final normalized =
        medicines.map((m) => normalizeMedicine(Map<String, dynamic>.from(m))).toList();
    return normalized.where((m) =>
      m['name'].toString().toLowerCase().contains(query.toLowerCase()) ||
      m['brand'].toString().toLowerCase().contains(query.toLowerCase()) ||
      m['genericName'].toString().toLowerCase().contains(query.toLowerCase())
    ).toList();
  }

  static Future<Map<String, dynamic>?> getMedicineById(String id) async {
    final medicines = await getMedicines();
    for (final medicine in medicines) {
      final item = Map<String, dynamic>.from(medicine as Map);
      if (item['id']?.toString() == id) {
        return item;
      }
    }
    return null;
  }

  static Future<List<dynamic>> getCategories() async {
    final data = await loadData();
    return data['categories'] as List;
  }

  static Future<List<dynamic>> getBanners() async {
    final data = await loadData();
    return data['banners'] as List;
  }

  static Future<List<dynamic>> getHealthTips() async {
    final data = await loadData();
    return data['healthTips'] as List;
  }
}
