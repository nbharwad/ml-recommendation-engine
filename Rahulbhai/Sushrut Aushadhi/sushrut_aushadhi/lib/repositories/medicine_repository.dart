import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/medicine_model.dart';
import '../models/category_model.dart';
import '../core/di/service_providers.dart';
import '../services/firestore_service.dart';

class MedicineRepository {
  final FirestoreService _firestoreService;

  MedicineRepository(this._firestoreService);

  static Provider<MedicineRepository> get provider => Provider<MedicineRepository>((ref) {
    return MedicineRepository(ref.read(firestoreServiceProvider));
  });

  Stream<List<MedicineModel>> getMedicinesStream({String? category}) {
    return _firestoreService.getMedicinesStream(category: category);
  }

  Future<MedicineModel?> getMedicineById(String id) async {
    return _firestoreService.getMedicineById(id);
  }

  Future<List<MedicineModel>> searchMedicines(String query) async {
    return _firestoreService.searchMedicines(query);
  }

  Stream<List<CategoryModel>> getCategoriesStream() {
    return _firestoreService.getCategoriesStream();
  }

  Future<void> addMedicine(MedicineModel medicine) async {
    return _firestoreService.addMedicine(medicine);
  }
}