import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/user_model.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';

final authServiceProvider = Provider<AuthService>((ref) {
  return AuthService();
});

final firestoreServiceProvider = Provider<FirestoreService>((ref) {
  return FirestoreService();
});

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
