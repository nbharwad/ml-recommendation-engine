import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/order_model.dart';
import '../services/firestore_service.dart';
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

final allOrdersProvider = StreamProvider<List<OrderModel>>((ref) async* {
  final firestoreService = ref.watch(firestoreServiceProvider);
  final roleAsync = ref.watch(roleProvider);
  
  if (roleAsync.valueOrNull != 'admin') {
    yield [];
    return;
  }
  
  yield* firestoreService.getAllOrders();
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

// ── Admin Orders Pagination ────────────────────────────────────────────────

class AdminOrdersPageState {
  final List<OrderModel> orders;
  final DocumentSnapshot? lastDoc;
  final bool isLoadingMore;
  final bool hasMore;
  final bool isInitialLoading;
  final Object? error;

  const AdminOrdersPageState({
    this.orders = const [],
    this.lastDoc,
    this.isLoadingMore = false,
    this.hasMore = true,
    this.isInitialLoading = true,
    this.error,
  });

  AdminOrdersPageState copyWith({
    List<OrderModel>? orders,
    DocumentSnapshot? lastDoc,
    bool? isLoadingMore,
    bool? hasMore,
    bool? isInitialLoading,
    Object? error,
  }) {
    return AdminOrdersPageState(
      orders: orders ?? this.orders,
      lastDoc: lastDoc ?? this.lastDoc,
      isLoadingMore: isLoadingMore ?? this.isLoadingMore,
      hasMore: hasMore ?? this.hasMore,
      isInitialLoading: isInitialLoading ?? this.isInitialLoading,
      error: error ?? this.error,
    );
  }
}

class AdminOrdersPageNotifier extends StateNotifier<AdminOrdersPageState> {
  final FirestoreService _firestoreService;
  static const int _pageSize = 20;
  int _loadId = 0; // guards against stale appends after refresh

  // Constructor does NOT auto-load — the provider triggers loadFirstPage()
  // once the admin JWT token is confirmed via roleProvider. This prevents a
  // permission-denied race condition when the widget builds before the token
  // refresh completes.
  AdminOrdersPageNotifier(this._firestoreService)
      : super(const AdminOrdersPageState());

  Future<void> loadFirstPage() async {
    final id = ++_loadId;
    state = const AdminOrdersPageState(); // reset to initial loading state
    try {
      final result =
          await _firestoreService.getOrdersPaginated(limit: _pageSize);
      if (_loadId != id) return; // stale — a newer refresh started
      state = state.copyWith(
        isInitialLoading: false,
        orders: result.orders,
        lastDoc: result.lastDoc,
        hasMore: result.orders.length == _pageSize,
      );
    } catch (e) {
      if (_loadId != id) return;
      state = state.copyWith(isInitialLoading: false, error: e);
    }
  }

  Future<void> loadNextPage() async {
    if (state.isLoadingMore || !state.hasMore) return;
    final id = _loadId;
    state = state.copyWith(isLoadingMore: true);
    try {
      final result = await _firestoreService.getOrdersPaginated(
          lastDoc: state.lastDoc, limit: _pageSize);
      if (_loadId != id) return;
      state = state.copyWith(
        isLoadingMore: false,
        orders: [...state.orders, ...result.orders],
        lastDoc: result.lastDoc ?? state.lastDoc,
        hasMore: result.orders.length == _pageSize,
      );
    } catch (_) {
      if (_loadId != id) return;
      state = state.copyWith(isLoadingMore: false);
    }
  }

  Future<void> refresh() => loadFirstPage();
}

final adminOrdersPageProvider =
    StateNotifierProvider<AdminOrdersPageNotifier, AdminOrdersPageState>((ref) {
  final notifier = AdminOrdersPageNotifier(ref.watch(firestoreServiceProvider));
  // Gate the first load on the admin role being confirmed from the refreshed JWT.
  // This prevents a permission-denied error when the widget builds before
  // getIdTokenResult(true) has completed and the token still lacks the role claim.
  ref.listen<AsyncValue<String?>>(roleProvider, (previous, next) {
    if (next.valueOrNull == 'admin') {
      notifier.loadFirstPage();
    }
  });
  return notifier;
});
