import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';
import '../models/notification_model.dart';

class NotificationNotifier extends StateNotifier<List<AppNotification>> {
  NotificationNotifier() : super([]) {
    _loadFromStorage();
  }

  static const String _key = 'app_notifications';
  static const int _maxNotifications = 20;

  Future<void> _loadFromStorage() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_key);
    if (data != null) {
      final list = jsonDecode(data) as List;
      state = list.map((e) => AppNotification.fromMap(e as Map<String, dynamic>)).toList();
    }
  }

  Future<void> _saveToStorage() async {
    final prefs = await SharedPreferences.getInstance();
    final data = jsonEncode(state.map((n) => n.toMap()).toList());
    await prefs.setString(_key, data);
  }

  Future<void> addNotification({
    required String title,
    required String body,
    required String type,
    String? orderId,
  }) async {
    final notification = AppNotification(
      id: const Uuid().v4(),
      title: title,
      body: body,
      type: type,
      orderId: orderId,
      isRead: false,
      createdAt: DateTime.now(),
    );

    state = [notification, ...state].take(_maxNotifications).toList();
    await _saveToStorage();
  }

  Future<void> markAsRead(String id) async {
    state = state.map((n) => n.id == id ? n.copyWith(isRead: true) : n).toList();
    await _saveToStorage();
  }

  Future<void> markAllAsRead() async {
    state = state.map((n) => n.copyWith(isRead: true)).toList();
    await _saveToStorage();
  }

  Future<void> clearAll() async {
    state = [];
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }

  int get unreadCount => state.where((n) => !n.isRead).length;
}

final notificationProvider = StateNotifierProvider<NotificationNotifier, List<AppNotification>>((ref) {
  return NotificationNotifier();
});

final unreadCountProvider = Provider<int>((ref) {
  final notifications = ref.watch(notificationProvider);
  return notifications.where((n) => !n.isRead).length;
});
