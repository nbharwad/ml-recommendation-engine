import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/lab_service.dart';
import '../models/lab_order_model.dart';
import 'auth_provider.dart';
import 'firebase_providers.dart';

final labServiceProvider = Provider<LabService>((ref) {
  final firestore = ref.read(firebaseFirestoreProvider);
  return LabService(firestore: firestore);
});

final userLabOrdersProvider = StreamProvider<List<LabOrderModel>>((ref) {
  final authState = ref.watch(authStateProvider);
  final labService = ref.watch(labServiceProvider);

  final user = authState.valueOrNull;
  if (user == null) {
    return Stream.value([]);
  }

  return labService.getUserLabOrders(user.uid);
});

final allLabOrdersProvider = StreamProvider<List<LabOrderModel>>((ref) {
  final labService = ref.watch(labServiceProvider);
  return labService.getAllLabOrders();
});

final labOrderProvider = StreamProvider.family<LabOrderModel?, String>((ref, orderId) {
  final labService = ref.watch(labServiceProvider);
  return labService.getLabOrderStream(orderId);
});

final labTestsProvider = FutureProvider<List<LabTestModel>>((ref) {
  print("=== LAB PROVIDER: Creating new provider instance ===");
  final labService = ref.watch(labServiceProvider);
  final future = labService.getLabTests();
  print("=== LAB PROVIDER: got future: $future ===");
  return future.then((tests) {
    print("=== LAB PROVIDER: Resolved with ${tests.length} tests ===");
    return tests;
  });
});

final todayLabSummaryProvider = FutureProvider<Map<String, dynamic>>((ref) {
  final labService = ref.watch(labServiceProvider);
  return labService.getTodayLabSummary();
});