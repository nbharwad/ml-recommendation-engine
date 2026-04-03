import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/order_model.dart';
import '../core/di/service_providers.dart';
import '../services/firestore_service.dart';

class OrderRepository {
  final FirestoreService _firestoreService;

  OrderRepository(this._firestoreService);

  static Provider<OrderRepository> get provider => Provider<OrderRepository>((ref) {
    return OrderRepository(ref.read(firestoreServiceProvider));
  });

  String placeOrderId() => _firestoreService.placeOrderId();

  Future<String> placeOrder(OrderModel order) async {
    return _firestoreService.placeOrder(order);
  }

  Stream<List<OrderModel>> getUserOrders(String userId) {
    return _firestoreService.getUserOrders(userId);
  }

  Stream<List<OrderModel>> getAllOrders() {
    return _firestoreService.getAllOrders();
  }

  Stream<List<OrderModel>> getOrdersByStatus(String status) {
    return _firestoreService.getOrdersByStatus(status);
  }

  Future<OrderModel?> getOrderById(String orderId) async {
    return _firestoreService.getOrderById(orderId);
  }

  Future<void> updateOrderStatus(String orderId, String status) async {
    return _firestoreService.updateOrderStatus(orderId, status);
  }
}