import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/order_model.dart';
import 'auth_provider.dart';
import '../core/di/service_providers.dart';

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

class PaginatedOrdersState {
  final List<OrderModel> orders;
  final DocumentSnapshot? lastDoc;
  final bool isLoading;
  final bool hasMore;
  final String? error;

  const PaginatedOrdersState({
    this.orders = const [],
    this.lastDoc,
    this.isLoading = false,
    this.hasMore = true,
    this.error,
  });

  PaginatedOrdersState copyWith({
    List<OrderModel>? orders,
    DocumentSnapshot? lastDoc,
    bool? isLoading,
    bool? hasMore,
    String? error,
  }) {
    return PaginatedOrdersState(
      orders: orders ?? this.orders,
      lastDoc: lastDoc ?? this.lastDoc,
      isLoading: isLoading ?? this.isLoading,
      hasMore: hasMore ?? this.hasMore,
      error: error,
    );
  }
}

class PaginatedOrdersNotifier extends StateNotifier<PaginatedOrdersState> {
  final Ref _ref;
  static const int _pageSize = 20;

  PaginatedOrdersNotifier(this._ref) : super(const PaginatedOrdersState()) {
    loadInitialOrders();
  }

  Future<void> loadInitialOrders() async {
    if (state.isLoading) return;
    state = state.copyWith(isLoading: true, error: null);

    try {
      final firestoreService = _ref.read(firestoreServiceProvider);
      final result = await firestoreService.getOrdersPaginated(limit: _pageSize);

      state = state.copyWith(
        orders: result.orders,
        lastDoc: result.lastDoc,
        isLoading: false,
        hasMore: result.orders.length >= _pageSize,
      );
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
    }
  }

  Future<void> loadMoreOrders() async {
    if (state.isLoading || !state.hasMore || state.lastDoc == null) return;
    state = state.copyWith(isLoading: true, error: null);

    try {
      final firestoreService = _ref.read(firestoreServiceProvider);
      final result = await firestoreService.getOrdersPaginated(
        lastDoc: state.lastDoc,
        limit: _pageSize,
      );

      state = state.copyWith(
        orders: [...state.orders, ...result.orders],
        lastDoc: result.lastDoc,
        isLoading: false,
        hasMore: result.orders.length >= _pageSize,
      );
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
    }
  }

  Future<void> refresh() async {
    state = const PaginatedOrdersState();
    await loadInitialOrders();
  }
}

final paginatedOrdersProvider = StateNotifierProvider<PaginatedOrdersNotifier, PaginatedOrdersState>((ref) {
  return PaginatedOrdersNotifier(ref);
});
