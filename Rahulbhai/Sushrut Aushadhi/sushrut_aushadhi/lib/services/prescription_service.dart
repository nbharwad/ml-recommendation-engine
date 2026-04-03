import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:path_provider/path_provider.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:uuid/uuid.dart';

import '../models/prescription_model.dart';
import '../core/utils/app_logger.dart';
import 'rate_limit_service.dart';

class PrescriptionService {
  final FirebaseFirestore _db;
  final FirebaseStorage _storage;
  final ImagePicker _picker;
  final Uuid _uuid;
  final RateLimitService _rateLimit;

  PrescriptionService({
    FirebaseFirestore? firestore,
    FirebaseStorage? storage,
    ImagePicker? picker,
    RateLimitService? rateLimitService,
  })  : _db = firestore ?? FirebaseFirestore.instance,
        _storage = storage ?? FirebaseStorage.instance,
        _picker = picker ?? ImagePicker(),
        _uuid = const Uuid(),
        _rateLimit = rateLimitService ?? RateLimitService();

  static const List<String> allowedImageTypes = ['jpg', 'jpeg', 'png', 'webp'];
  static const int maxFileSizeMB = 10;
  static const int maxDailyUploads = 5;

  // ── REQUEST CAMERA PERMISSION ──────────────────────────
  Future<PermissionStatus> requestCameraPermission() async {
    var status = await Permission.camera.status;
    if (status.isDenied) {
      status = await Permission.camera.request();
    }
    return status;
  }

  // ── REQUEST STORAGE/GALLERY PERMISSION ────────────────
  Future<PermissionStatus> requestGalleryPermission() async {
    PermissionStatus status;

    if (Platform.isAndroid) {
      // Android 13+ uses READ_MEDIA_IMAGES
      final androidInfo = await _getAndroidVersion();
      if (androidInfo >= 33) {
        status = await Permission.photos.status;
        if (status.isDenied) {
          status = await Permission.photos.request();
        }
      } else {
        // Android 12 and below
        status = await Permission.storage.status;
        if (status.isDenied) {
          status = await Permission.storage.request();
        }
      }
    } else {
      status = await Permission.photos.status;
      if (status.isDenied) {
        status = await Permission.photos.request();
      }
    }
    return status;
  }

  // Get Android SDK version
  Future<int> _getAndroidVersion() async {
    try {
      if (Platform.isAndroid) {
        final version = Platform.operatingSystemVersion;
        final match = RegExp(r'(\d+)').firstMatch(version);
        return int.tryParse(match?.group(1) ?? '0') ?? 0;
      }
    } catch (e) {
      debugPrint('Version check error: $e');
    }
    return 0;
  }

  // ── PICK IMAGE FROM CAMERA ─────────────────────────────
  Future<File?> pickFromCamera() async {
    try {
      // Request permission
      final status = await requestCameraPermission();

      if (status.isPermanentlyDenied) {
        throw PermissionException(
            'Camera permission permanently denied. '
            'Please enable in app settings.');
      }

      if (!status.isGranted) {
        throw PermissionException(
            'Camera permission denied.');
      }

      // Pick image
      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 85,
        preferredCameraDevice: CameraDevice.rear,
      );

      if (photo == null) return null;
      return File(photo.path);

    } on PermissionException {
      rethrow;
    } catch (e) {
      debugPrint('Camera error: $e');
      throw Exception('Could not open camera: $e');
    }
  }

  // ── PICK IMAGE FROM GALLERY ────────────────────────────
  Future<File?> pickFromGallery() async {
    try {
      // Request permission
      final status = await requestGalleryPermission();

      if (status.isPermanentlyDenied) {
        throw PermissionException(
            'Storage permission permanently denied. '
            'Please enable in app settings.');
      }

      // Note: On Android 13+, image_picker handles
      // permissions internally — proceed even if
      // status shows denied
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 85,
      );

      if (image == null) return null;
      return File(image.path);

    } on PermissionException {
      rethrow;
    } catch (e) {
      debugPrint('Gallery error: $e');
      throw Exception('Could not open gallery: $e');
    }
  }

  // ── COMPRESS IMAGE ─────────────────────────────────────
  Future<File> compressImage(File file) async {
    try {
      final dir = await getTemporaryDirectory();
      final targetPath =
          '${dir.path}/prescription_'
          '${DateTime.now().millisecondsSinceEpoch}.jpg';

      final result =
          await FlutterImageCompress.compressAndGetFile(
        file.absolute.path,
        targetPath,
        quality: 75,
        minWidth: 800,
        minHeight: 800,
        format: CompressFormat.jpeg,
      );

      if (result == null) return file;
      return File(result.path);
    } catch (e) {
      debugPrint('Compress error: $e');
      return file; // Return original if compress fails
    }
  }

  // ── VALIDATE FILE ──────────────────────────────────────
  bool isValidSize(File file) {
    final bytes = file.lengthSync();
    final mb = bytes / (1024 * 1024);
    return mb <= maxFileSizeMB.toDouble();
  }

  bool isValidImageType(String filePath) {
    final extension = filePath.split('.').last.toLowerCase();
    return allowedImageTypes.contains(extension);
  }

  // ── UPLOAD TO FIREBASE STORAGE ─────────────────────────
  Future<String> uploadPrescription({
    required String userId,
    required File imageFile,
    required Function(double) onProgress,
  }) async {
    // Validate file type
    if (!isValidImageType(imageFile.path)) {
      throw Exception('Invalid file type. Please upload JPG, PNG, or WebP images.');
    }

    // Check daily limit
    await _rateLimit.checkPrescriptionRateLimit(userId);

    // Compress image
    File fileToUpload = imageFile;
    try {
      fileToUpload = await compressImage(imageFile);
    } catch (e) {
      debugPrint('Compression failed, using original');
    }

    // Validate size
    if (!isValidSize(fileToUpload)) {
      throw Exception(
          'File is too large. Please use a '
          'clearer, smaller image.');
    }

    // Use UUID for secure filename
    final fileName = '${_uuid.v4()}.jpg';
    final ref = _storage.ref(
        'prescriptions/$userId/$fileName');

    // Upload with progress
    final uploadTask = ref.putFile(
      fileToUpload,
      SettableMetadata(contentType: 'image/jpeg'),
    );

    uploadTask.snapshotEvents.listen((snapshot) {
      if (snapshot.totalBytes > 0) {
        final progress = snapshot.bytesTransferred /
            snapshot.totalBytes;
        onProgress(progress.clamp(0.0, 1.0));
      }
    });

    await uploadTask;
    return await ref.getDownloadURL();
  }

  // ── SAVE PRESCRIPTION TO FIRESTORE ────────────────────
  Future<String> savePrescription({
    required String userId,
    required String userPhone,
    required String userName,
    required String imageUrl,
    String? orderId,
  }) async {
    final doc = _db.collection('prescriptions').doc();
    final now = DateTime.now();

    final sanitizedName = _sanitizeString(userName);
    final sanitizedPhone = _sanitizeString(userPhone);

    await doc.set({
      'userId': userId,
      'userPhone': sanitizedPhone,
      'userName': sanitizedName,
      'imageUrl': imageUrl,
      'status': 'pending',
      'orderId': orderId,
      'notes': null,
      'createdAt': Timestamp.fromDate(now),
      'updatedAt': Timestamp.fromDate(now),
    });

    AppLogger.info("Prescription uploaded: ${doc.id}", tag: "Prescription");
    AppLogger.logCustomKey("prescription_id", doc.id);
    
    return doc.id;
  }

  String _sanitizeString(String input) {
    return input
        .trim()
        .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), '')
        .replaceAll(RegExp(r'<[^>]*>'), '');
  }

  // ── GET USER PRESCRIPTIONS ─────────────────────────────
  Stream<List<PrescriptionModel>> getUserPrescriptions(
      String userId) {
    return _db
        .collection('prescriptions')
        .where('userId', isEqualTo: userId)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((doc) => PrescriptionModel
                .fromFirestore(
                    doc.data(), doc.id))
            .toList());
  }

  // ── GET ALL PRESCRIPTIONS (Admin) ──────────────────────
  Stream<List<PrescriptionModel>> getAllPrescriptions({
    String? status,
  }) {
    Query query = _db
        .collection('prescriptions')
        .orderBy('createdAt', descending: true);

    if (status != null && status != 'all') {
      query = query.where('status',
          isEqualTo: status);
    }

    return query.snapshots().map((snap) => snap.docs
        .map((doc) => PrescriptionModel.fromFirestore(
            doc.data() as Map<String, dynamic>,
            doc.id))
        .toList());
  }

  // ── GET PRESCRIPTIONS BY STATUS ────────────────────────
  Stream<List<PrescriptionModel>> getPrescriptionsByStatus(String status) {
    return _db
        .collection('prescriptions')
        .where('status', isEqualTo: status)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((doc) => PrescriptionModel.fromFirestore(
                doc.data() as Map<String, dynamic>,
                doc.id))
            .toList());
  }

  // ── UPDATE STATUS (Admin) ──────────────────────────────
  Future<void> updatePrescriptionStatus({
    required String prescriptionId,
    required String status,
    String? notes,
  }) async {
    await _db
        .collection('prescriptions')
        .doc(prescriptionId)
        .update({
      'status': status,
      'notes': notes,
      'updatedAt':
          FieldValue.serverTimestamp(),
    });
  }
}

// Custom exception for permissions
class PermissionException implements Exception {
  final String message;
  PermissionException(this.message);
  String toString() => message;
}