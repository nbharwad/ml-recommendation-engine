import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/medicine_model.dart';
import 'auth_provider.dart';
import '../core/di/service_providers.dart';

final selectedCategoryProvider = StateProvider<String>((ref) => 'all');

final medicinesProvider = StreamProvider.family<List<MedicineModel>, String>(
  (ref, category) {
    final firestoreService = ref.read(firestoreServiceProvider);
    return firestoreService.getMedicinesStream(category: category);
  },
);

final searchQueryProvider = StateProvider<String>((ref) => '');

final searchResultsProvider = FutureProvider.family<List<MedicineModel>, String>(
  (ref, query) async {
    if (query.isEmpty) return [];
    final firestoreService = ref.read(firestoreServiceProvider);
    return firestoreService.searchMedicines(query);
  },
);

final medicineByIdProvider = FutureProvider.family<MedicineModel?, String>((ref, medicineId) async {
  final firestoreService = ref.watch(firestoreServiceProvider);
  return await firestoreService.getMedicineById(medicineId);
});
