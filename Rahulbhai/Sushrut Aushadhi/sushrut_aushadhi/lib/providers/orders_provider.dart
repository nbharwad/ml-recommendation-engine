import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/order_model.dart';
import 'auth_provider.dart';

final ordersProvider = StreamProvider<List<OrderModel>>((ref) {
  final firestoreService = ref.watch(firestoreServiceProvider);
  final uid = ref.watch(authStateProvider).valueOrNull?.uid;

  if (uid == null) {
    return Stream.value([]);
  }

  return firestoreService.getUserOrders(uid);
});

final allOrdersProvider = StreamProvider<List<OrderModel>>((ref) {
  final firestoreService = ref.watch(firestoreServiceProvider);
  return firestoreService.getAllOrders();
});

final orderByStatusProvider = StreamProvider.family<List<OrderModel>, String>((ref, status) {
  final firestoreService = ref.watch(firestoreServiceProvider);
  return firestoreService.getOrdersByStatus(status);
});

final orderByIdProvider = FutureProvider.family<OrderModel?, String>((ref, orderId) async {
  final firestoreService = ref.watch(firestoreServiceProvider);
  return await firestoreService.getOrderById(orderId);
});

final selectedStatusProvider = StateProvider<String?>((ref) => null);
