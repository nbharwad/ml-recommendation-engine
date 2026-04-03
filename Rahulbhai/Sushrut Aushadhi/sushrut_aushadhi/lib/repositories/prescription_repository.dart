import 'dart:io';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/prescription_model.dart';
import '../core/di/service_providers.dart';
import '../services/prescription_service.dart';

class PrescriptionRepository {
  final PrescriptionService _prescriptionService;

  PrescriptionRepository(this._prescriptionService);

  static Provider<PrescriptionRepository> get provider => Provider<PrescriptionRepository>((ref) {
    return PrescriptionRepository(ref.read(prescriptionServiceProvider));
  });

  Future<File?> pickFromCamera() async {
    return _prescriptionService.pickFromCamera();
  }

  Future<File?> pickFromGallery() async {
    return _prescriptionService.pickFromGallery();
  }

  Future<File> compressImage(File file) async {
    return _prescriptionService.compressImage(file);
  }

  bool isValidSize(File file) {
    return _prescriptionService.isValidSize(file);
  }

  bool isValidImageType(String filePath) {
    return _prescriptionService.isValidImageType(filePath);
  }

  Future<String> uploadPrescription({
    required String userId,
    required File imageFile,
    required Function(double) onProgress,
  }) async {
    return _prescriptionService.uploadPrescription(
      userId: userId,
      imageFile: imageFile,
      onProgress: onProgress,
    );
  }

  Future<String> savePrescription({
    required String userId,
    required String userPhone,
    required String userName,
    required String imageUrl,
    String? orderId,
  }) async {
    return _prescriptionService.savePrescription(
      userId: userId,
      userPhone: userPhone,
      userName: userName,
      imageUrl: imageUrl,
      orderId: orderId,
    );
  }

  Stream<List<PrescriptionModel>> getUserPrescriptions(String userId) {
    return _prescriptionService.getUserPrescriptions(userId);
  }

  Stream<List<PrescriptionModel>> getAllPrescriptions({String? status}) {
    return _prescriptionService.getAllPrescriptions(status: status);
  }

  Stream<List<PrescriptionModel>> getPrescriptionsByStatus(String status) {
    return _prescriptionService.getPrescriptionsByStatus(status);
  }

  Future<void> updatePrescriptionStatus({
    required String prescriptionId,
    required String status,
    String? notes,
  }) async {
    return _prescriptionService.updatePrescriptionStatus(
      prescriptionId: prescriptionId,
      status: status,
      notes: notes,
    );
  }
}