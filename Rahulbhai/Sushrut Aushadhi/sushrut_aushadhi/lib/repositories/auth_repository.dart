import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../models/user_model.dart';
import '../core/di/service_providers.dart';
import '../services/auth_service.dart';

class AuthRepository {
  final AuthService _authService;

  AuthRepository(this._authService);

  static Provider<AuthRepository> get provider => Provider<AuthRepository>((ref) {
    return AuthRepository(ref.read(authServiceProvider));
  });

  Stream<User?> get authStateChanges => _authService.authStateChanges;

  User? get currentUser => _authService.currentUser;

  Future<void> sendOtp({
    required String phoneNumber,
    required Function(String) onCodeSent,
    required Function(String) onError,
    Function(UserCredential)? onVerificationCompleted,
    Function(String)? onTimeout,
  }) async {
    return _authService.sendOtp(
      phoneNumber: phoneNumber,
      onCodeSent: onCodeSent,
      onError: onError,
      onVerificationCompleted: onVerificationCompleted,
      onTimeout: onTimeout,
    );
  }

  Future<UserModel?> verifyOtp({
    required String verificationId,
    required String smsCode,
  }) async {
    return _authService.verifyOtp(
      verificationId: verificationId,
      smsCode: smsCode,
    );
  }

  Future<UserModel> getOrCreateUser(String uid, String phone) async {
    return _authService.getOrCreateUser(uid, phone);
  }

  Future<void> signOut() async {
    return _authService.signOut();
  }
}