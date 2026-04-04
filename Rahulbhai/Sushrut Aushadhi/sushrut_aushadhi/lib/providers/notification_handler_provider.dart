import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../core/di/service_providers.dart';

class NotificationHandlerNotifier extends StateNotifier<bool> {
  final Ref _ref;
  bool _isInitialized = false;

  NotificationHandlerNotifier(this._ref) : super(false);

  Future<void> initialize(GoRouter router) async {
    if (_isInitialized) return;
    _isInitialized = true;

    final notificationService = _ref.read(notificationServiceProvider);
    await notificationService.initialize();

    notificationService.setOrderNotificationTapCallback((String orderId) {
      router.push('/order/$orderId');
    });

    await notificationService.setupMessageHandlers();
    state = true;
  }
}

final notificationHandlerProvider = StateNotifierProvider<NotificationHandlerNotifier, bool>((ref) {
  return NotificationHandlerNotifier(ref);
});
