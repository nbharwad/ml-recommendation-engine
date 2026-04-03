import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/firebase_providers.dart';
import '../../services/auth_service.dart';
import '../../services/firestore_service.dart';
import '../../services/prescription_service.dart';
import '../../services/rate_limit_service.dart';
import '../../services/storage_service.dart';
import '../../services/search_history_service.dart';
import '../../services/notification_service.dart';
import '../../services/connectivity_service.dart';
import '../../services/whatsapp_service.dart';
import '../../services/lab_service.dart';

final authServiceProvider = Provider<AuthService>((ref) {
  final auth = ref.read(firebaseAuthProvider);
  final firestore = ref.read(firebaseFirestoreProvider);
  return AuthService(auth: auth, firestore: firestore);
});

final firestoreServiceProvider = Provider<FirestoreService>((ref) {
  final firestore = ref.read(firebaseFirestoreProvider);
  return FirestoreService(firestore: firestore);
});

final prescriptionServiceProvider = Provider<PrescriptionService>((ref) {
  final firestore = ref.read(firebaseFirestoreProvider);
  return PrescriptionService(firestore: firestore);
});

final rateLimitServiceProvider = Provider<RateLimitService>((ref) {
  return RateLimitService();
});

final storageServiceProvider = Provider<StorageService>((ref) {
  return StorageService();
});

final searchHistoryServiceProvider = Provider<SearchHistoryService>((ref) {
  return SearchHistoryService();
});

final notificationServiceProvider = Provider<NotificationService>((ref) {
  return NotificationService();
});

final connectivityServiceProvider = Provider<ConnectivityService>((ref) {
  return ConnectivityService();
});

final whatsappServiceProvider = Provider<WhatsAppService>((ref) {
  return WhatsAppService();
});

final labServiceProvider = Provider<LabService>((ref) {
  final firestore = ref.read(firebaseFirestoreProvider);
  return LabService(firestore: firestore);
});
