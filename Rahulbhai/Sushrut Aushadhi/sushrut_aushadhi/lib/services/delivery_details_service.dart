import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';

class DeliveryDetails {
  final String phone;
  final String address;
  final String pincode;

  const DeliveryDetails({
    required this.phone,
    required this.address,
    required this.pincode,
  });

  bool get isComplete =>
      phone.trim().length >= 10 &&
      address.trim().length >= 5 &&
      pincode.trim().length == 6;

  DeliveryDetails copyWith({
    String? phone,
    String? address,
    String? pincode,
  }) =>
      DeliveryDetails(
        phone: phone ?? this.phone,
        address: address ?? this.address,
        pincode: pincode ?? this.pincode,
      );
}

class DeliveryDetailsService {
  static Future<DeliveryDetails> getDetails() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        return const DeliveryDetails(
          phone: '',
          address: '',
          pincode: '',
        );
      }

      String phone = user.phoneNumber ?? '';

      final doc = await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .get();

      if (doc.exists && doc.data() != null) {
        final data = doc.data()!;

        if (phone.isEmpty) {
          phone = data['phone'] ?? '';
        }

        String address = '';
        String pincode = '';

        final deliveryAddress = data['deliveryAddress'];
        if (deliveryAddress is Map<String, dynamic>) {
          address = (deliveryAddress['line1'] ?? '').toString().trim();
          pincode = (deliveryAddress['pincode'] ?? '').toString().trim();
        }

        if (address.isEmpty) {
          address = (data['address'] ?? '').toString().trim();
        }
        if (pincode.isEmpty) {
          pincode = (data['pincode'] ?? '').toString().trim();
        }

        final details = DeliveryDetails(
          phone: phone,
          address: address,
          pincode: pincode,
        );

        debugPrint(
          'Delivery details loaded: phone=${details.phone}, address=${details.address}, pincode=${details.pincode}, isComplete=${details.isComplete}',
        );

        return details;
      }

      return DeliveryDetails(
        phone: phone,
        address: '',
        pincode: '',
      );
    } catch (e) {
      debugPrint('Error getting delivery details: $e');
      return const DeliveryDetails(
        phone: '',
        address: '',
        pincode: '',
      );
    }
  }

  static Future<void> saveDetails({
    required String phone,
    required String address,
    required String pincode,
  }) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      throw Exception('Not logged in');
    }

    phone = phone.trim();
    address = address.trim();
    pincode = pincode.trim();

    if (phone.length < 10) {
      throw Exception('Enter a valid phone number');
    }
    if (address.length < 5) {
      throw Exception('Enter a complete delivery address');
    }
    if (pincode.length != 6) {
      throw Exception('Enter a valid 6-digit pincode');
    }

    try {
      final docRef =
          FirebaseFirestore.instance.collection('users').doc(user.uid);
      final existingDoc = await docRef.get();

      Map<String, dynamic> existingDeliveryAddress = {};
      if (existingDoc.exists) {
        final data = existingDoc.data();
        if (data != null && data['deliveryAddress'] is Map<String, dynamic>) {
          existingDeliveryAddress =
              Map<String, dynamic>.from(data['deliveryAddress'] as Map);
        }
      }

      final deliveryAddress = {
        'line1': address,
        'line2': existingDeliveryAddress['line2'],
        'city': existingDeliveryAddress['city'],
        'state': existingDeliveryAddress['state'],
        'pincode': pincode,
      };

      final updateData = {
        'phone': phone,
        'address': address,
        'pincode': pincode,
        'deliveryAddress': deliveryAddress,
        'updatedAt': FieldValue.serverTimestamp(),
      };

      debugPrint('Saving delivery details: $updateData');

      await docRef.set(updateData, SetOptions(merge: true));

      debugPrint('Delivery details saved successfully');
    } catch (e) {
      debugPrint('Error saving delivery details: $e');
      rethrow;
    }
  }
}
