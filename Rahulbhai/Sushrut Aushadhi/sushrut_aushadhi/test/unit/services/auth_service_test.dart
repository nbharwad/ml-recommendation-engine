import 'package:flutter_test/flutter_test.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import 'package:sushrut_aushadhi/models/user_model.dart';
import 'package:sushrut_aushadhi/models/order_model.dart';
import 'package:sushrut_aushadhi/models/delivery_address.dart';

void main() {
  group('UserModel Tests', () {
    test('fromFirestore parses user data correctly', () {
      final data = {
        'name': 'Test User',
        'phone': '+911234567890',
        'isAdmin': false,
        'address': {
          'line1': '123 Test Street',
          'line2': '',
          'city': 'Test City',
          'state': 'Test State',
          'pincode': '123456',
        },
        'createdAt': Timestamp.now(),
      };

      final user = UserModel.fromFirestore(data, 'user-123');
      expect(user.uid, equals('user-123'));
      expect(user.name, equals('Test User'));
      expect(user.phone, equals('+911234567890'));
      expect(user.isAdmin, isFalse);
    });

    test('fromFirestore handles isAdmin field correctly', () {
      final data = {
        'name': 'Admin User',
        'phone': '+911234567890',
        'isAdmin': true,
        'createdAt': Timestamp.now(),
      };

      final user = UserModel.fromFirestore(data, 'admin-123');
      expect(user.isAdmin, isTrue);
    });

    test('fromFirestore handles missing optional fields', () {
      final data = {
        'name': 'Test User',
        'phone': '+911234567890',
        'createdAt': Timestamp.now(),
      };

      final user = UserModel.fromFirestore(data, 'user-123');
      expect(user.name, equals('Test User'));
      expect(user.fcmToken, isNull);
      expect(user.deliveryAddress, isNull);
    });

    test('toMap serializes user correctly', () {
      final user = UserModel(
        uid: 'user-123',
        name: 'Test User',
        phone: '+911234567890',
        isAdmin: false,
        createdAt: DateTime.now(),
      );

      final map = user.toMap();
      expect(map['uid'], equals('user-123'));
      expect(map['name'], equals('Test User'));
      expect(map['phone'], equals('+911234567890'));
    });
  });

  group('OrderModel Tests', () {
    test('fromFirestore parses order data correctly', () {
      final data = {
        'userId': 'user-123',
        'userPhone': '+911234567890',
        'userName': 'Test User',
        'deliveryAddress': {
          'line1': '123 Test Street',
          'line2': '',
          'city': 'Test City',
          'state': 'Test State',
          'pincode': '123456',
        },
        'items': [
          {
            'medicineId': 'med-001',
            'medicineName': 'Test Medicine',
            'price': 99.99,
            'quantity': 2,
            'subtotal': 199.98,
          },
        ],
        'totalAmount': 199.98,
        'status': 'pending',
        'paymentMethod': 'cod',
        'paymentStatus': 'pending',
        'createdAt': Timestamp.now(),
        'updatedAt': Timestamp.now(),
      };

      final order = OrderModel.fromFirestore(data, 'order-001');
      expect(order.orderId, equals('order-001'));
      expect(order.userId, equals('user-123'));
      expect(order.totalAmount, equals(199.98));
      expect(order.status, equals(OrderStatus.pending));
    });

    test('OrderStatus.fromString parses status correctly', () {
      expect(OrderStatus.fromString('pending'), equals(OrderStatus.pending));
      expect(OrderStatus.fromString('delivered'), equals(OrderStatus.delivered));
      expect(OrderStatus.fromString('cancelled'), equals(OrderStatus.cancelled));
    });

    test('OrderStatus.fromString handles unknown status', () {
      final status = OrderStatus.fromString('unknown_status');
      expect(status, equals(OrderStatus.pending));
    });
  });

  group('DeliveryAddress Tests', () {
    test('toMap serializes address correctly', () {
      final address = DeliveryAddress(
        line1: '123 Test Street',
        line2: 'Apt 4',
        city: 'Test City',
        state: 'Test State',
        pincode: '123456',
      );

      final map = address.toMap();
      expect(map['line1'], equals('123 Test Street'));
      expect(map['line2'], equals('Apt 4'));
      expect(map['city'], equals('Test City'));
      expect(map['state'], equals('Test State'));
      expect(map['pincode'], equals('123456'));
    });

    test('toDisplayString formats address correctly', () {
      final address = DeliveryAddress(
        line1: '123 Test Street',
        line2: 'Apt 4',
        city: 'Test City',
        state: 'Test State',
        pincode: '123456',
      );

      final displayString = address.toDisplayString();
      expect(displayString, contains('123 Test Street'));
      expect(displayString, contains('Test City'));
    });
  });
}