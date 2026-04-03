import 'package:flutter_test/flutter_test.dart';
import 'package:sushrut_aushadhi/models/user_model.dart';
import 'package:sushrut_aushadhi/models/delivery_address.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

void main() {
  group('UserModel', () {
    group('fromFirestore', () {
      test('should create UserModel from valid Firestore data', () {
        final data = {
          'name': 'John Doe',
          'phone': '+911234567890',
          'address': '123 Main Street, City',
          'pincode': '123456',
          'isAdmin': false,
          'fcmToken': 'test-token-123',
          'createdAt': Timestamp.fromDate(DateTime(2024, 1, 1)),
        };

        final user = UserModel.fromFirestore(data, 'user-123');

        expect(user.uid, equals('user-123'));
        expect(user.name, equals('John Doe'));
        expect(user.phone, equals('+911234567890'));
        expect(user.address, equals('123 Main Street, City'));
        expect(user.pincode, equals('123456'));
        expect(user.isAdmin, isFalse);
        expect(user.fcmToken, equals('test-token-123'));
      });

      test('should handle admin flag', () {
        final data = {
          'name': 'Admin User',
          'phone': '+911234567890',
          'isAdmin': true,
        };

        final user = UserModel.fromFirestore(data, 'admin-123');

        expect(user.isAdmin, isTrue);
      });

      test('should handle missing optional fields with defaults', () {
        final data = <String, dynamic>{
          'name': 'Test User',
          'phone': '+911234567890',
        };

        final user = UserModel.fromFirestore(data, 'user-123');

        expect(user.address, equals(''));
        expect(user.pincode, equals(''));
        expect(user.isAdmin, isFalse);
        expect(user.fcmToken, isNull);
      });
    });

    group('toMap', () {
      test('should convert UserModel to Map with Timestamp', () {
        final user = UserModel(
          uid: 'user-123',
          name: 'John Doe',
          phone: '+911234567890',
          deliveryAddress: DeliveryAddress(
            line1: '123 Main Street',
            city: 'City',
            state: 'State',
            pincode: '123456',
          ),
          isAdmin: false,
          fcmToken: 'test-token',
          createdAt: DateTime(2024, 1, 1),
        );

        final map = user.toMap();

        expect(map['uid'], equals('user-123'));
        expect(map['name'], equals('John Doe'));
        expect(map['phone'], equals('+911234567890'));
        expect(map['address'], isA<Map<String, dynamic>>());
        expect((map['address'] as Map)['line1'], equals('123 Main Street'));
        expect((map['address'] as Map)['pincode'], equals('123456'));
        expect(map['isAdmin'], isFalse);
        expect(map['fcmToken'], equals('test-token'));
        expect(map['createdAt'], isA<Timestamp>());
      });

      test('should be reversible (fromFirestore -> toMap)', () {
        final data = {
          'name': 'Test User',
          'phone': '+911234567890',
          'address': 'Test Address',
          'pincode': '123456',
          'isAdmin': true,
        };

        final user = UserModel.fromFirestore(data, 'test-uid');
        final map = user.toMap();

        expect(map['name'], equals(data['name']));
        expect(map['phone'], equals(data['phone']));
        expect(map['address'], equals(data['address']));
        expect(map['pincode'], equals(data['pincode']));
        expect(map['isAdmin'], equals(data['isAdmin']));
      });
    });

    group('copyWith', () {
      test('should create copy with updated name', () {
        final user = UserModel(
          uid: 'user-123',
          name: 'John',
          phone: '+911234567890',
          createdAt: DateTime.now(),
        );

        final updatedUser = user.copyWith(name: 'Jane');

        expect(updatedUser.name, equals('Jane'));
        expect(updatedUser.uid, equals('user-123'));
        expect(updatedUser.phone, equals('+911234567890'));
      });

      test('should create copy with updated address', () {
        final user = UserModel(
          uid: 'user-123',
          name: 'John',
          phone: '+911234567890',
          deliveryAddress: DeliveryAddress(line1: 'Old Address', city: '', state: '', pincode: ''),
          createdAt: DateTime.now(),
        );

        final updatedUser = user.copyWith(
          deliveryAddress: DeliveryAddress(line1: 'New Address', city: '', state: '', pincode: ''),
        );

        expect(updatedUser.address, equals('New Address'));
        expect(user.address, equals('Old Address'));
      });

      test('should create copy with updated isAdmin', () {
        final user = UserModel(
          uid: 'user-123',
          name: 'User',
          phone: '+911234567890',
          isAdmin: false,
          createdAt: DateTime.now(),
        );

        final adminUser = user.copyWith(isAdmin: true);

        expect(adminUser.isAdmin, isTrue);
        expect(user.isAdmin, isFalse);
      });

      test('should preserve all fields when no arguments provided', () {
        final user = UserModel(
          uid: 'user-123',
          name: 'John',
          phone: '+911234567890',
          deliveryAddress: DeliveryAddress(line1: 'Test Address', city: 'City', state: 'State', pincode: '123456'),
          isAdmin: true,
          fcmToken: 'token',
          createdAt: DateTime(2024, 1, 1),
        );

        final copy = user.copyWith();

        expect(copy.uid, equals(user.uid));
        expect(copy.name, equals(user.name));
        expect(copy.phone, equals(user.phone));
        expect(copy.address, equals(user.address));
        expect(copy.pincode, equals(user.pincode));
        expect(copy.isAdmin, equals(user.isAdmin));
        expect(copy.fcmToken, equals(user.fcmToken));
      });
    });
  });
}
