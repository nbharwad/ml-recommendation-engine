import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_sign_in/google_sign_in.dart';

import '../models/user_model.dart';
import '../core/utils/app_logger.dart';
import '../providers/firebase_providers.dart';

class AuthService {
  final FirebaseAuth _auth;
  final FirebaseFirestore _db;

  AuthService({FirebaseAuth? auth, FirebaseFirestore? firestore})
      : _auth = auth ?? FirebaseAuth.instance,
        _db = firestore ?? FirebaseFirestore.instance;

  Stream<User?> get authStateChanges => _auth.authStateChanges();

  User? get currentUser => _auth.currentUser;

  Future<void> sendOtp({
    required String phoneNumber,
    required Function(String) onCodeSent,
    required Function(String) onError,
    Function(UserCredential)? onVerificationCompleted,
    Function(String)? onTimeout,
  }) async {
    try {
      await _auth.verifyPhoneNumber(
        phoneNumber: phoneNumber,
        verificationCompleted: (PhoneAuthCredential credential) async {
          final userCredential = await _auth.signInWithCredential(credential);
          if (userCredential.user != null) {
            await getOrCreateUser(
              userCredential.user!.uid,
              userCredential.user!.phoneNumber ?? phoneNumber,
            );
          }
          onVerificationCompleted?.call(userCredential);
        },
        verificationFailed: (FirebaseAuthException e) {
          onError(e.message ?? 'Verification failed');
        },
        codeSent: (String verificationId, int? resendToken) {
          onCodeSent(verificationId);
        },
        codeAutoRetrievalTimeout: (String verificationId) {
          onTimeout?.call(verificationId);
        },
      );
    } catch (e) {
      onError(e.toString());
    }
  }

  Future<UserModel?> verifyOtp({
    required String verificationId,
    required String smsCode,
  }) async {
    final credential = PhoneAuthProvider.credential(
      verificationId: verificationId,
      smsCode: smsCode,
    );
    final userCredential = await _auth.signInWithCredential(credential);

    if (userCredential.user != null) {
      final user = await getOrCreateUser(
        userCredential.user!.uid,
        userCredential.user!.phoneNumber ?? '',
      );
      
      // Set user context for Crashlytics
      FirebaseCrashlytics.instance.setUserIdentifier(user.uid);
      AppLogger.info("User logged in", tag: "Auth");
      
      return user;
    }
    return null;
  }

  Future<UserModel> getOrCreateUser(String uid, String phone, {String name = '', String email = ''}) async {
    final docRef = _db.collection('users').doc(uid);
    final doc = await docRef.get();

    if (doc.exists) {
      return UserModel.fromFirestore(doc.data()!, uid);
    }

    final newUser = UserModel(
      uid: uid,
      name: name,
      phone: phone,
      email: email,
      createdAt: DateTime.now(),
    );

    await docRef.set(newUser.toMap());
    return newUser;
  }

  Future<void> signOut() async {
    await _auth.signOut();
  }

  Future<UserModel?> signInWithGoogle() async {
    try {
      final GoogleSignInAccount? googleUser = await GoogleSignIn().signIn();
      if (googleUser == null) return null;

      final GoogleSignInAuthentication googleAuth = await googleUser.authentication;

      final credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      final userCredential = await _auth.signInWithCredential(credential);
      final user = userCredential.user;
      if (user == null) return null;

      return await getOrCreateUser(
        user.uid,
        user.phoneNumber ?? '',
        name: user.displayName ?? '',
        email: user.email ?? '',
      );
    } catch (e) {
      throw Exception('Google sign in failed: $e');
    }
  }

  Future<UserModel?> signUpWithEmail({
    required String email,
    required String password,
    required String name,
  }) async {
    try {
      final userCredential = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      final user = userCredential.user;
      if (user == null) return null;

      await user.updateDisplayName(name);

      return await getOrCreateUser(
        user.uid,
        '',
        name: name,
        email: email,
      );
    } catch (e) {
      if (e is FirebaseAuthException) {
        switch (e.code) {
          case 'email-already-in-use':
            throw Exception('This email is already registered. Please login.');
          case 'weak-password':
            throw Exception('Password is too weak. Use at least 6 characters.');
          case 'invalid-email':
            throw Exception('Invalid email address.');
          default:
            throw Exception('Sign up failed: ${e.message}');
        }
      }
      throw Exception('Sign up failed: $e');
    }
  }

  Future<UserModel?> signInWithEmail({
    required String email,
    required String password,
  }) async {
    try {
      final userCredential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      final user = userCredential.user;
      if (user == null) return null;

      return await getOrCreateUser(
        user.uid,
        user.phoneNumber ?? '',
        name: user.displayName ?? '',
        email: email,
      );
    } catch (e) {
      if (e is FirebaseAuthException) {
        switch (e.code) {
          case 'user-not-found':
            throw Exception('No account found. Please sign up first.');
          case 'wrong-password':
            throw Exception('Wrong password. Please try again.');
          case 'invalid-email':
            throw Exception('Invalid email address.');
          case 'user-disabled':
            throw Exception('This account has been disabled.');
          default:
            throw Exception('Login failed: ${e.message}');
        }
      }
      throw Exception('Login failed: $e');
    }
  }

  Future<void> sendPasswordResetEmail(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
    } catch (e) {
      throw Exception('Could not send reset email: $e');
    }
  }
}
