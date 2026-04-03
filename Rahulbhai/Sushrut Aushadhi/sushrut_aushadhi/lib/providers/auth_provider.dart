import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/user_model.dart';
import '../core/di/service_providers.dart';

final authStateProvider = StreamProvider<User?>((ref) {
  final authService = ref.watch(authServiceProvider);
  return authService.authStateChanges;
});

final currentUserProvider = StreamProvider<UserModel?>((ref) {
  final authState = ref.watch(authStateProvider);
  final firestoreService = ref.watch(firestoreServiceProvider);

  final user = authState.valueOrNull;
  if (user == null) {
    return Stream.value(null);
  }

  return firestoreService.getUserStream(user.uid);
});

/// Reads the user's role from their JWT custom claims.
/// Force-refreshes the token to get the latest claims set by Cloud Functions.
final roleProvider = FutureProvider<String?>((ref) async {
  final authState = ref.watch(authStateProvider);
  final user = authState.valueOrNull;
  if (user == null) return null;

  final idTokenResult = await user.getIdTokenResult(true);
  return idTokenResult.claims?['role'] as String?;
});

final isAdminProvider = Provider<bool>((ref) {
  final roleAsync = ref.watch(roleProvider);
  return roleAsync.maybeWhen(
    data: (role) => role == 'admin',
    orElse: () => false,
  );
});

/// True only after the auth stream has emitted at least once AND the role
/// future has settled. Used by the splash screen to prevent premature routing.
final authReadyProvider = Provider<bool>((ref) {
  final authAsync = ref.watch(authStateProvider);
  if (authAsync is AsyncLoading) return false;
  final user = authAsync.valueOrNull;
  if (user == null) return true;
  final roleAsync = ref.watch(roleProvider);
  return roleAsync is! AsyncLoading;
});
