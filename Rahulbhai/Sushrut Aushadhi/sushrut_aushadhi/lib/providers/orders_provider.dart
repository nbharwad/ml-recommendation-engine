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

final orderByStatusProvider =
    StreamProvider.family<List<OrderModel>, String>((ref, status) {
  final firestoreService = ref.watch(firestoreServiceProvider);
  return firestoreService.getOrdersByStatus(status);
});

final orderByIdProvider =
    FutureProvider.family<OrderModel?, String>((ref, orderId) async {
  final firestoreService = ref.watch(firestoreServiceProvider);
  return await firestoreService.getOrderById(orderId);
});

final selectedStatusProvider = StateProvider<String?>((ref) => null);

// ── Admin Orders Pagination ────────────────────────────────────────────────

enum PageErrorType {
  none,
  permissionDenied,
  queryPrecondition,
  network,
  authResolution,
  unknown,
}

class AdminOrdersPageState {
  final List<OrderModel> orders;
  final DocumentSnapshot? lastDoc;
  final bool isLoadingMore;
  final bool hasMore;
  final bool isInitialLoading;
  final AdminAuthStatus? authStatus;
  final PageErrorType errorType;
  final String? errorMessage;
  final PageErrorType paginationError;
  final String? paginationErrorMessage;

  const AdminOrdersPageState({
    this.orders = const [],
    this.lastDoc,
    this.isLoadingMore = false,
    this.hasMore = true,
    this.isInitialLoading = true,
    this.authStatus,
    this.errorType = PageErrorType.none,
    this.errorMessage,
    this.paginationError = PageErrorType.none,
    this.paginationErrorMessage,
  });

  AdminOrdersPageState copyWith({
    List<OrderModel>? orders,
    DocumentSnapshot? lastDoc,
    bool? isLoadingMore,
    bool? hasMore,
    bool? isInitialLoading,
    AdminAuthStatus? authStatus,
    PageErrorType? errorType,
    String? errorMessage,
    PageErrorType? paginationError,
    String? paginationErrorMessage,
  }) {
    return AdminOrdersPageState(
      orders: orders ?? this.orders,
      lastDoc: lastDoc ?? this.lastDoc,
      isLoadingMore: isLoadingMore ?? this.isLoadingMore,
      hasMore: hasMore ?? this.hasMore,
      isInitialLoading: isInitialLoading ?? this.isInitialLoading,
      authStatus: authStatus ?? this.authStatus,
      errorType: errorType ?? this.errorType,
      errorMessage: errorMessage ?? this.errorMessage,
      paginationError: paginationError ?? this.paginationError,
      paginationErrorMessage:
          paginationErrorMessage ?? this.paginationErrorMessage,
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
        errorType: PageErrorType.none,
        errorMessage: null,
        paginationError: PageErrorType.none,
        paginationErrorMessage: null,
      );
    } catch (e) {
      if (_loadId != id) return;
      final errorType = _classifyError(e);
      state = state.copyWith(
        isInitialLoading: false,
        errorType: errorType,
        errorMessage: e.toString(),
      );
    }
  }

  PageErrorType _classifyError(dynamic error) {
    if (error is FirebaseException) {
      final code = error.code;
      final message = error.message?.toLowerCase() ?? '';

      if (code == 'permission-denied') return PageErrorType.permissionDenied;
      if (code == 'failed-precondition') {
        if (message.contains('index') || message.contains('query')) {
          return PageErrorType.queryPrecondition;
        }
        return PageErrorType.queryPrecondition;
      }
      if (code == 'unavailable') return PageErrorType.network;
    }
    return PageErrorType.unknown;
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
        paginationError: PageErrorType.none,
        paginationErrorMessage: null,
      );
    } catch (e) {
      if (_loadId != id) return;
      final errorType = _classifyError(e);
      state = state.copyWith(
        isLoadingMore: false,
        paginationError: errorType,
        paginationErrorMessage: e.toString(),
      );
    }
  }

  Future<void> refresh() => loadFirstPage();

  void setAuthLoading() {
    state = state.copyWith(
      isInitialLoading: true,
      authStatus: AdminAuthStatus.loading,
    );
  }

  void setAuthDenied() {
    state = AdminOrdersPageState(
      authStatus: AdminAuthStatus.notAdmin,
      isInitialLoading: false,
      hasMore: false,
    );
  }

  void setAuthError(String? errorMessage) {
    state = AdminOrdersPageState(
      authStatus: AdminAuthStatus.error,
      isInitialLoading: false,
      hasMore: false,
      errorType: PageErrorType.authResolution,
      errorMessage: errorMessage,
    );
  }
}

final adminOrdersPageProvider =
    StateNotifierProvider<AdminOrdersPageNotifier, AdminOrdersPageState>((ref) {
  final notifier = AdminOrdersPageNotifier(ref.watch(firestoreServiceProvider));

  // Gate loads on admin auth state - react to all states explicitly
  ref.listen<AsyncValue<AdminAuthState>>(adminAuthStateProvider,
      (previous, next) {
    final authState = next.valueOrNull;

    if (authState == null || authState.status == AdminAuthStatus.loading) {
      // Loading - set loading state, do NOT fetch
      notifier.setAuthLoading();
    } else if (authState.status == AdminAuthStatus.notAdmin) {
      // Not admin - clear data, set denied state
      notifier.setAuthDenied();
    } else if (authState.status == AdminAuthStatus.error) {
      // Auth error - clear data, set auth error
      notifier.setAuthError(authState.error?.toString());
    } else if (authState.status == AdminAuthStatus.admin) {
      // Admin - trigger fetch
      notifier.loadFirstPage();
    }
  });

  return notifier;
});
