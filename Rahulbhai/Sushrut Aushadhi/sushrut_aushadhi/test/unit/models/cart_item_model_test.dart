import 'package:flutter_test/flutter_test.dart';
import 'package:sushrut_aushadhi/models/cart_item_model.dart';
import 'package:sushrut_aushadhi/models/medicine_model.dart';

void main() {
  group('CartItem', () {
    group('constructor', () {
      test('should create CartItem with default quantity of 1', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine);

        expect(cartItem.quantity, equals(1));
        expect(cartItem.medicine, equals(medicine));
      });

      test('should create CartItem with custom quantity', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine, quantity: 5);

        expect(cartItem.quantity, equals(5));
      });
    });

    group('subtotal', () {
      test('should calculate subtotal correctly', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine, quantity: 3);

        expect(cartItem.subtotal, equals(90.0));
      });

      test('should return 0 when quantity is 0', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine, quantity: 0);

        expect(cartItem.subtotal, equals(0.0));
      });

      test('should handle decimal prices', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 99.99,
          mrp: 149.99,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine, quantity: 2);

        expect(cartItem.subtotal, closeTo(199.98, 0.01));
      });
    });

    group('copyWith', () {
      test('should create copy with updated quantity', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine, quantity: 2);
        final updatedItem = cartItem.copyWith(quantity: 5);

        expect(updatedItem.quantity, equals(5));
        expect(updatedItem.medicine, equals(medicine));
        expect(cartItem.quantity, equals(2)); // original unchanged
      });

      test('should create copy with updated medicine', () {
        final medicine1 = MedicineModel(
          id: 'med-001',
          name: 'Medicine 1',
          category: 'General',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final medicine2 = MedicineModel(
          id: 'med-002',
          name: 'Medicine 2',
          category: 'General',
          price: 50.0,
          mrp: 75.0,
          stock: 50,
        );

        final cartItem = CartItem(medicine: medicine1, quantity: 2);
        final updatedItem = cartItem.copyWith(medicine: medicine2);

        expect(updatedItem.medicine.id, equals('med-002'));
        expect(updatedItem.quantity, equals(2));
      });

      test('should preserve original when no arguments', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
        );

        final cartItem = CartItem(medicine: medicine, quantity: 3);
        final copy = cartItem.copyWith();

        expect(copy.medicine, equals(cartItem.medicine));
        expect(copy.quantity, equals(cartItem.quantity));
      });
    });
  });
}
