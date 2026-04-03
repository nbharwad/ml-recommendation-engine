import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:uuid/uuid.dart';
import '../models/lab_order_model.dart';
import '../core/utils/app_logger.dart';
import '../services/notification_service.dart';

class LabService {
  final FirebaseFirestore _db;
  final _uuid = const Uuid();
  final NotificationService _notificationService;

  LabService({
    FirebaseFirestore? firestore,
    NotificationService? notificationService,
  })  : _db = firestore ?? FirebaseFirestore.instance,
        _notificationService = notificationService ?? NotificationService();

  String generateLabOrderId() => _uuid.v4();

  Future<Map<String, dynamic>> getTodayLabSummary() async {
    try {
      final now = DateTime.now();
      final startOfDay = DateTime(now.year, now.month, now.day);
      final endOfDay = startOfDay.add(const Duration(days: 1));

      final snapshot = await _db
          .collection('labOrders')
          .where('createdAt', isGreaterThanOrEqualTo: Timestamp.fromDate(startOfDay))
          .where('createdAt', isLessThan: Timestamp.fromDate(endOfDay))
          .get();

      final orders = snapshot.docs
          .map((doc) => LabOrderModel.fromFirestore(doc.data(), doc.id))
          .toList();

      final pending = orders.where((o) => o.status == LabOrderStatus.pending).length;
      final sampleCollected = orders.where((o) => o.status == LabOrderStatus.sampleCollected).length;
      final processing = orders.where((o) => o.status == LabOrderStatus.processing).length;
      final completed = orders.where((o) => o.status == LabOrderStatus.completed).length;
      final cancelled = orders.where((o) => o.status == LabOrderStatus.cancelled).length;

      final paidOrders = orders.where((o) => o.paymentStatus == 'paid').toList();
      final unpaidOrders = orders.where((o) => o.paymentStatus != 'paid').toList();

      final totalRevenue = paidOrders.fold<double>(0, (sum, o) => sum + o.totalAmount);
      final pendingRevenue = unpaidOrders.fold<double>(0, (sum, o) => sum + o.totalAmount);

      return {
        'totalOrders': orders.length,
        'pending': pending,
        'sampleCollected': sampleCollected,
        'processing': processing,
        'completed': completed,
        'cancelled': cancelled,
        'paidOrders': paidOrders.length,
        'unpaidOrders': unpaidOrders.length,
        'totalRevenue': totalRevenue,
        'pendingRevenue': pendingRevenue,
        'orders': orders,
      };
    } catch (e) {
      AppLogger.error("Error getting today's lab summary: $e", tag: "LabOrder");
      rethrow;
    }
  }

  Future<String> placeLabOrder(LabOrderModel order) async {
    // Input validation
    if (order.tests.isEmpty || order.tests.length > 20) {
      throw Exception('Invalid test count: must be between 1 and 20');
    }
    if (order.totalAmount <= 0 || order.totalAmount > 50000) {
      throw Exception('Invalid total amount: must be between 1 and 50000');
    }
    if (order.userId.isEmpty || order.userPhone.isEmpty || order.userName.isEmpty) {
      throw Exception('Missing required user information');
    }

    try {
      final orderId = order.orderId.isNotEmpty ? order.orderId : generateLabOrderId();

      final orderWithId = LabOrderModel(
        orderId: orderId,
        userId: order.userId,
        userPhone: order.userPhone,
        userName: order.userName,
        homeCollectionAddress: order.homeCollectionAddress,
        tests: order.tests,
        totalAmount: order.totalAmount,
        status: order.status,
        paymentMethod: order.paymentMethod,
        paymentStatus: order.paymentStatus,
        notes: order.notes,
        createdAt: order.createdAt,
        updatedAt: order.updatedAt,
        statusHistory: order.statusHistory,
      );

      await _db.collection('labOrders').doc(orderId).set(orderWithId.toMap());
      
      AppLogger.info("Lab order placed: $orderId", tag: "LabOrder");
      
      return orderId;
    } catch (e) {
      AppLogger.error("Error placing lab order: $e", tag: "LabOrder");
      rethrow;
    }
  }

  Stream<List<LabOrderModel>> getUserLabOrders(String userId) {
    return _db
        .collection('labOrders')
        .where('userId', isEqualTo: userId)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snapshot) {
      return snapshot.docs
          .map((doc) => LabOrderModel.fromFirestore(doc.data(), doc.id))
          .toList();
    });
  }

  Stream<List<LabOrderModel>> getAllLabOrders() {
    return _db
        .collection('labOrders')
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snapshot) {
      return snapshot.docs
          .map((doc) => LabOrderModel.fromFirestore(doc.data(), doc.id))
          .toList();
    });
  }

  Future<LabOrderModel?> getLabOrderById(String orderId) async {
    final doc = await _db.collection('labOrders').doc(orderId).get();
    if (doc.exists) {
      return LabOrderModel.fromFirestore(doc.data()!, doc.id);
    }
    return null;
  }

  Stream<LabOrderModel?> getLabOrderStream(String orderId) {
    return _db.collection('labOrders').doc(orderId).snapshots().map((doc) {
      if (doc.exists) {
        return LabOrderModel.fromFirestore(doc.data()!, doc.id);
      }
      return null;
    });
  }

  Future<void> updateLabOrderStatus(String orderId, LabOrderStatus status, {String? note, String? updatedBy}) async {
    final newHistoryEntry = LabStatusHistoryEntry(
      status: status.name,
      timestamp: DateTime.now(),
      updatedBy: updatedBy,
      note: note,
    );

    final Map<String, dynamic> updateData = {
      'status': status.name,
      'updatedAt': FieldValue.serverTimestamp(),
      'statusHistory': FieldValue.arrayUnion([newHistoryEntry.toMap()]),
    };

    if (status == LabOrderStatus.completed) {
      updateData['completedAt'] = FieldValue.serverTimestamp();
    }

    await _db.collection('labOrders').doc(orderId).update(updateData);

    if (status == LabOrderStatus.completed) {
      await _sendCompletionNotification(orderId);
    }
  }

  Future<void> _sendCompletionNotification(String orderId) async {
    try {
      final order = await getLabOrderById(orderId);
      if (order == null) return;

      final shortId = orderId.length >= 8 ? orderId.substring(0, 8) : orderId;
      final title = 'Lab Results Ready';
      final body = 'Your lab order #SA-LB-${shortId.toUpperCase()} is complete. View your results in the app.';

      await _notificationService.addLocalNotification(
        title: title,
        body: body,
        type: 'lab_completed',
        orderId: orderId,
      );

      await _db.collection('notifications').doc(orderId).set({
        'userId': order.userId,
        'title': title,
        'body': body,
        'type': 'lab_completed',
        'orderId': orderId,
        'isRead': false,
        'createdAt': FieldValue.serverTimestamp(),
      });

      AppLogger.info("Completion notification sent for order: $orderId", tag: "LabOrder");
    } catch (e) {
      AppLogger.error("Error sending completion notification: $e", tag: "LabOrder");
    }
  }

  Future<List<LabTestModel>> getLabTests() async {
    try {
      print("=== LAB SERVICE: Starting fetch from lab_tests collection ===");

      // Force server timestamp to bypass cache
      final snapshot = await _db.collection('lab_tests')
          .limit(10)
          .get(const GetOptions(source: Source.server));

      print("=== LAB SERVICE: Query completed, docs count: ${snapshot.docs.length} ===");

      // Parse documents individually to catch per-document errors
      final tests = <LabTestModel>[];
      for (var doc in snapshot.docs) {
        try {
          print("=== PARSING DOC: ${doc.id}, Data: ${doc.data()} ===");
          final test = LabTestModel.fromFirestore(doc.data(), doc.id);
          tests.add(test);
          print("=== SUCCESS: Parsed ${test.name} ===");
        } catch (e, st) {
          print("=== PARSE ERROR for ${doc.id}: $e ===");
          print("=== STACK: $st ===");
          AppLogger.error("Failed to parse lab test ${doc.id}: $e", tag: "LabService");
          // Continue parsing other documents
        }
      }

      print("=== LAB SERVICE: Successfully parsed ${tests.length} tests ===");
      return tests;
    } catch (e, st) {
      print("=== LAB SERVICE ERROR: $e ===");
      print("=== LAB SERVICE STACK: $st ===");
      AppLogger.error("Error fetching lab tests: $e", tag: "LabService");
      rethrow;
    }
  }

  Future<void> updatePaymentStatus(String orderId, String paymentStatus) async {
    try {
      final updateData = <String, dynamic>{
        'paymentStatus': paymentStatus,
        'updatedAt': FieldValue.serverTimestamp(),
      };

      if (paymentStatus == 'paid') {
        updateData['paidAt'] = FieldValue.serverTimestamp();
      }

      await _db.collection('labOrders').doc(orderId).update(updateData);
      AppLogger.info("Payment status updated to $paymentStatus for order: $orderId", tag: "LabOrder");
    } catch (e) {
      AppLogger.error("Error updating payment status: $e", tag: "LabOrder");
      rethrow;
    }
  }

  Future<String> uploadLabResult(String orderId, String filePath, String fileName) async {
    try {
      final storageRef = FirebaseStorage.instance.ref().child('lab_results').child(orderId).child(fileName);
      
      final uploadTask = storageRef.putFile(
        File(filePath),
        SettableMetadata(contentType: 'application/pdf'),
      );
      
      final snapshot = await uploadTask;
      final downloadUrl = await snapshot.ref.getDownloadURL();
      
      await _db.collection('labOrders').doc(orderId).update({
        'labResultUrl': downloadUrl,
        'resultUploaded': true,
        'updatedAt': FieldValue.serverTimestamp(),
      });
      
      AppLogger.info("Lab result uploaded for order: $orderId", tag: "LabOrder");
      return downloadUrl;
    } catch (e) {
      AppLogger.error("Error uploading lab result: $e", tag: "LabOrder");
      rethrow;
    }
  }
}