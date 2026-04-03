import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/medicine_model.dart';
import '../models/cart_item_model.dart';
import '../models/order_model.dart';
import '../core/di/service_providers.dart';

class CartNotifier extends StateNotifier<List<CartItem>> {
  final Ref _ref;
  CartNotifier(this._ref) : super([]) {
    _loadCart();
  }

  static const String _cartKey = 'saved_cart';

  Future<void> _loadCart() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final data = prefs.getString(_cartKey);
      if (data == null) return;

      final List decoded = jsonDecode(data);

      final List<CartItem> items = [];
      for (final item in decoded) {
        final medicine = MedicineModel(
          id: item['id'],
          name: item['name'],
          price: (item['price'] as num).toDouble(),
          mrp: (item['mrp'] as num).toDouble(),
          category: item['category'] ?? 'other',
          stock: item['stock'] ?? 50,
          requiresPrescription: item['requiresPrescription'] ?? false,
          manufacturer: item['manufacturer'] ?? '',
          unit: item['unit'] ?? 'strip',
          isActive: true,
        );
        items.add(CartItem(medicine: medicine, quantity: item['quantity']));
      }
      state = items;
    } catch (e) {
      state = [];
    }
  }

  Future<void> _saveCart() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final data = jsonEncode(state.map((item) => {
        'id': item.medicine.id,
        'name': item.medicine.name,
        'price': item.medicine.price,
        'mrp': item.medicine.mrp,
        'category': item.medicine.category,
        'stock': item.medicine.stock,
        'requiresPrescription': item.medicine.requiresPrescription,
        'manufacturer': item.medicine.manufacturer,
        'unit': item.medicine.unit,
        'quantity': item.quantity,
      }).toList());
      await prefs.setString(_cartKey, data);
    } catch (e) {}
  }

  Future<void> reorderFromOrder(OrderModel order) async {
    state = [];
    
    final firestoreService = _ref.read(firestoreServiceProvider);
    int added = 0;
    int outOfStock = 0;
    int notFound = 0;

    for (final item in order.items) {
      final medicine = await firestoreService.getMedicineById(item.medicineId);
      
      if (medicine == null) {
        notFound++;
        continue;
      }
      
      if (medicine.stock < 1) {
        outOfStock++;
        continue;
      }
      
      addItem(medicine, quantity: item.quantity);
      added++;
    }

    return;
  }

  void addItem(MedicineModel medicine, {int quantity = 1}) {
    final existingIndex = state.indexWhere((item) => item.medicine.id == medicine.id);
    
    if (existingIndex != -1) {
      final updatedItems = [...state];
      updatedItems[existingIndex].quantity += quantity;
      state = updatedItems;
    } else {
      state = [...state, CartItem(medicine: medicine, quantity: quantity)];
    }
    _saveCart();
  }

  void removeItem(String medicineId) {
    state = state.where((item) => item.medicine.id != medicineId).toList();
    _saveCart();
  }

  void updateQuantity(String medicineId, int quantity) {
    if (quantity <= 0) {
      removeItem(medicineId);
      return;
    }

    state = state.map((item) {
      if (item.medicine.id == medicineId) {
        return item.copyWith(quantity: quantity);
      }
      return item;
    }).toList();
    _saveCart();
  }

  void clearCart() {
    state = [];
    _saveCart();
  }

  double get totalAmount {
    return state.fold(0, (sum, item) => sum + item.subtotal);
  }

  int get itemCount {
    return state.fold(0, (sum, item) => sum + item.quantity);
  }

  bool get requiresPrescription {
    return state.any((item) => item.medicine.requiresPrescription);
  }

  List<CartItem> get items => state;
}

final cartProvider = StateNotifierProvider<CartNotifier, List<CartItem>>((ref) {
  return CartNotifier(ref);
});

final cartTotalProvider = Provider<double>((ref) {
  final cart = ref.watch(cartProvider);
  return cart.fold(0, (sum, item) => sum + item.subtotal);
});

final cartItemCountProvider = Provider<int>((ref) {
  final cart = ref.watch(cartProvider);
  return cart.fold(0, (sum, item) => sum + item.quantity);
});

final cartRequiresPrescriptionProvider = Provider<bool>((ref) {
  final cart = ref.watch(cartProvider);
  return cart.any((item) => item.medicine.requiresPrescription);
});
