import 'package:cloud_firestore/cloud_firestore.dart';

class DeliveryAddress {
  final String line1;
  final String? line2;
  final String city;
  final String state;
  final String pincode;

  DeliveryAddress({
    required this.line1,
    this.line2,
    required this.city,
    required this.state,
    required this.pincode,
  });

  factory DeliveryAddress.fromMap(Map<String, dynamic> data) {
    return DeliveryAddress(
      line1: data['line1'] ?? '',
      line2: data['line2'],
      city: data['city'] ?? '',
      state: data['state'] ?? '',
      pincode: data['pincode'] ?? '',
    );
  }

  factory DeliveryAddress.fromLegacyString(String address) {
    return DeliveryAddress(
      line1: address,
      line2: null,
      city: '',
      state: '',
      pincode: '',
    );
  }

  factory DeliveryAddress.fromFirestoreData(dynamic data) {
    if (data is String) {
      return DeliveryAddress.fromLegacyString(data);
    }
    if (data is Map<String, dynamic>) {
      return DeliveryAddress.fromMap(data);
    }
    return DeliveryAddress(
      line1: '',
      city: '',
      state: '',
      pincode: '',
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'line1': line1,
      if (line2 != null) 'line2': line2,
      'city': city,
      'state': state,
      'pincode': pincode,
    };
  }

  String toDisplayString() {
    final parts = <String>[];
    if (line1.isNotEmpty) parts.add(line1);
    if (line2 != null && line2!.isNotEmpty) parts.add(line2!);
    if (city.isNotEmpty) parts.add(city);
    if (state.isNotEmpty) parts.add(state);
    if (pincode.isNotEmpty) parts.add(pincode);
    return parts.join(', ');
  }

  bool get isEmpty => line1.isEmpty && city.isEmpty && pincode.isEmpty;

  DeliveryAddress copyWith({
    String? line1,
    String? line2,
    String? city,
    String? state,
    String? pincode,
  }) {
    return DeliveryAddress(
      line1: line1 ?? this.line1,
      line2: line2 ?? this.line2,
      city: city ?? this.city,
      state: state ?? this.state,
      pincode: pincode ?? this.pincode,
    );
  }
}