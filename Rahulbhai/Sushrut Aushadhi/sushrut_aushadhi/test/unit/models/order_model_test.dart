import 'package:flutter_test/flutter_test.dart';
import 'package:sushrut_aushadhi/models/order_model.dart';
import 'package:sushrut_aushadhi/models/delivery_address.dart';

void main() {
  group('OrderStatus', () {
    test('displayName returns correct names', () {
      expect(OrderStatus.pending.displayName, equals('Pending'));
      expect(OrderStatus.delivered.displayName, equals('Delivered'));
    });

    test('fromString parses valid status', () {
      expect(OrderStatus.fromString('pending'), equals(OrderStatus.pending));
      expect(OrderStatus.fromString('delivered'), equals(OrderStatus.delivered));
    });

    test('fromString returns pending for invalid', () {
      expect(OrderStatus.fromString('invalid'), equals(OrderStatus.pending));
    });
  });

  group('OrderItem', () {
    test('fromMap creates item correctly', () {
      final data = {
        'medicineId': 'med-001',
        'medicineName': 'Paracetamol',
        'price': 30.0,
        'quantity': 2,
        'subtotal': 60.0,
      };
      final item = OrderItem.fromMap(data);
      expect(item.medicineId, equals('med-001'));
      expect(item.quantity, equals(2));
    });

    test('toMap converts correctly', () {
      final item = OrderItem(
        medicineId: 'med-001',
        medicineName: 'Paracetamol',
        price: 30.0,
        quantity: 2,
        subtotal: 60.0,
      );
      final map = item.toMap();
      expect(map['medicineId'], equals('med-001'));
    });
  });

  group('OrderModel', () {
    test('fromFirestore creates order correctly', () {
      final data = {
        'userId': 'user-123',
        'userPhone': '+911234567890',
        'userName': 'Test User',
        'deliveryAddress': '123 Test St',
        'items': <Map<String, dynamic>>[],
        'totalAmount': 100.0,
        'status': 'pending',
        'paymentMethod': 'cod',
        'paymentStatus': 'pending',
      };
      final order = OrderModel.fromFirestore(data, 'order-001');
      expect(order.orderId, equals('order-001'));
      expect(order.totalAmount, equals(100.0));
    });

    test('itemCount calculates correctly', () {
      final order = OrderModel(
        orderId: 'order-001',
        userId: 'user-123',
        userPhone: '+911234567890',
        userName: 'Test User',
        deliveryAddress: DeliveryAddress(line1: 'Test', city: '', state: '', pincode: ''),
        items: [
          OrderItem(medicineId: '1', medicineName: 'M1', price: 10, quantity: 2, subtotal: 20),
          OrderItem(medicineId: '2', medicineName: 'M2', price: 15, quantity: 3, subtotal: 45),
        ],
        totalAmount: 65.0,
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
      );
      expect(order.itemCount, equals(5));
    });
  });
}
