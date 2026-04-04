import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/prescription_model.dart';
import '../repositories/prescription_repository.dart';
import 'auth_provider.dart';

final userPrescriptionsProvider = StreamProvider<List<PrescriptionModel>>((ref) {
  final uid = ref.watch(authStateProvider).valueOrNull?.uid;
  if (uid == null) {
    return Stream.value([]);
  }
  final repository = ref.watch(PrescriptionRepository.provider);
  return repository.getUserPrescriptions(uid);
});

final hasUploadedPrescriptionProvider = Provider<bool>((ref) {
  final prescriptionsAsync = ref.watch(userPrescriptionsProvider);
  return prescriptionsAsync.maybeWhen(
    data: (prescriptions) {
      return prescriptions.any((p) => p.status != PrescriptionStatus.rejected);
    },
    orElse: () => false,
  );
});

final pendingPrescriptionCountProvider = Provider<int>((ref) {
  final prescriptionsAsync = ref.watch(userPrescriptionsProvider);
  return prescriptionsAsync.maybeWhen(
    data: (prescriptions) {
      return prescriptions.where((p) => p.status == PrescriptionStatus.pending).length;
    },
    orElse: () => 0,
  );
});

final approvedPrescriptionCountProvider = Provider<int>((ref) {
  final prescriptionsAsync = ref.watch(userPrescriptionsProvider);
  return prescriptionsAsync.maybeWhen(
    data: (prescriptions) {
      return prescriptions.where((p) => p.status == PrescriptionStatus.approved).length;
    },
    orElse: () => 0,
  );
});
