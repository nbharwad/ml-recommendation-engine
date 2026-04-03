import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/user_model.dart';
import '../core/di/service_providers.dart';
import '../services/firestore_service.dart';

class UserRepository {
  final FirestoreService _firestoreService;

  UserRepository(this._firestoreService);

  static Provider<UserRepository> get provider => Provider<UserRepository>((ref) {
    return UserRepository(ref.read(firestoreServiceProvider));
  });

  Future<UserModel?> getUser(String uid) async {
    return _firestoreService.getUser(uid);
  }

  Stream<UserModel?> getUserStream(String uid) {
    return _firestoreService.getUserStream(uid);
  }

  Future<void> updateUser(String uid, Map<String, dynamic> data) async {
    return _firestoreService.updateUser(uid, data);
  }
}