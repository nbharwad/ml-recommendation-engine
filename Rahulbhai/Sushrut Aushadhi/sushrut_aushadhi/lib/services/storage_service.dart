import 'dart:io';

import 'package:firebase_storage/firebase_storage.dart';

class StorageService {
  final FirebaseStorage _storage = FirebaseStorage.instance;
  static const int _maxPrescriptionBytes = 5 * 1024 * 1024;

  Future<String> uploadPrescription({
    required String userId,
    required String orderId,
    required File imageFile,
  }) async {
    final fileSize = await imageFile.length();
    if (fileSize > _maxPrescriptionBytes) {
      throw Exception('Prescription image must be 5 MB or smaller.');
    }

    final ref = _storage
        .ref()
        .child('prescriptions')
        .child(userId)
        .child('$orderId.jpg');

    final uploadTask = ref.putFile(
      imageFile,
      SettableMetadata(contentType: 'image/jpeg'),
    );

    final snapshot = await uploadTask;
    return snapshot.ref.getDownloadURL();
  }

  Future<String> uploadMedicineImage({
    required String medicineId,
    required File imageFile,
  }) async {
    final ref = _storage.ref().child('medicines').child('$medicineId.jpg');

    final uploadTask = ref.putFile(
      imageFile,
      SettableMetadata(contentType: 'image/jpeg'),
    );

    final snapshot = await uploadTask;
    return snapshot.ref.getDownloadURL();
  }

  Future<void> deleteImage(String imageUrl) async {
    try {
      final ref = _storage.refFromURL(imageUrl);
      await ref.delete();
    } catch (_) {
      // Image might not exist.
    }
  }
}
