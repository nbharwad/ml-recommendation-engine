import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sushrut_aushadhi/models/medicine_model.dart';
import 'package:sushrut_aushadhi/providers/cart_provider.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  group('CartNotifier', () {
    late ProviderContainer container;

    setUp(() {
      container = ProviderContainer();
    });

    tearDown(() {
      container.dispose();
    });

    test('should add new item to cart', () {
      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Paracetamol',
        category: 'Pain Relief',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
      );

      cartNotifier.addItem(medicine, quantity: 2);

      final cart = container.read(cartProvider);
      expect(cart.length, equals(1));
      expect(cart[0].medicine.id, equals('med-001'));
      expect(cart[0].quantity, equals(2));
    });

    test('should increase quantity when adding existing item', () {
      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Paracetamol',
        category: 'Pain Relief',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
      );

      cartNotifier.addItem(medicine, quantity: 1);
      cartNotifier.addItem(medicine, quantity: 2);

      final cart = container.read(cartProvider);
      expect(cart.length, equals(1));
      expect(cart[0].quantity, equals(3));
    });

    test('should remove item from cart', () {
      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Paracetamol',
        category: 'Pain Relief',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
      );

      cartNotifier.addItem(medicine);
      cartNotifier.removeItem('med-001');

      final cart = container.read(cartProvider);
      expect(cart, isEmpty);
    });

    test('should update quantity of existing item', () {
      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Paracetamol',
        category: 'Pain Relief',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
      );

      cartNotifier.addItem(medicine, quantity: 1);
      cartNotifier.updateQuantity('med-001', 5);

      final cart = container.read(cartProvider);
      expect(cart[0].quantity, equals(5));
    });

    test('should clear all items from cart', () {
      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Paracetamol',
        category: 'Pain Relief',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
      );

      cartNotifier.addItem(medicine);
      cartNotifier.clearCart();

      final cart = container.read(cartProvider);
      expect(cart, isEmpty);
    });
  });

  group('cartTotalProvider', () {
    test('should calculate total amount correctly', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);

      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Paracetamol',
        category: 'Pain Relief',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
      );

      cartNotifier.addItem(medicine, quantity: 2);

      final total = container.read(cartTotalProvider);
      expect(total, equals(60.0));
    });

    test('should return 0 for empty cart', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);

      final total = container.read(cartTotalProvider);
      expect(total, equals(0.0));
    });
  });

  group('cartRequiresPrescriptionProvider', () {
    test('should return true when any item requires prescription', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);

      final cartNotifier = container.read(cartProvider.notifier);
      final medicine = MedicineModel(
        id: 'med-001',
        name: 'Rx Medicine',
        category: 'Rx',
        price: 30.0,
        mrp: 45.0,
        stock: 100,
        requiresPrescription: true,
      );

      cartNotifier.addItem(medicine);

      final requiresRx = container.read(cartRequiresPrescriptionProvider);
      expect(requiresRx, isTrue);
    });

    test('should return false for empty cart', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);

      final requiresRx = container.read(cartRequiresPrescriptionProvider);
      expect(requiresRx, isFalse);
    });
  });
}
