import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:uuid/uuid.dart';

import '../models/category_model.dart';
import '../models/medicine_model.dart';
import '../models/order_model.dart';
import '../models/user_model.dart';
import '../core/utils/app_logger.dart';
import 'local_data_service.dart';
import 'rate_limit_service.dart';

class FirestoreService {
  final FirebaseFirestore _db;
  final _uuid = const Uuid();
  late final RateLimitService _rateLimit;

  FirestoreService({FirebaseFirestore? firestore}) : _db = firestore ?? FirebaseFirestore.instance {
    _rateLimit = RateLimitService(firestore: _db);
  }

  String placeOrderId() => _uuid.v4();

  Stream<List<MedicineModel>> getMedicinesStream({String? category}) async* {
    Query<Map<String, dynamic>> query = _db
        .collection('medicines')
        .where('isActive', isEqualTo: true)
        .limit(50);

    if (category != null && category != 'all') {
      query = query.where('category', isEqualTo: category);
    }

    try {
      await for (final snapshot in query.snapshots()) {
        yield snapshot.docs
            .map((doc) =>
                MedicineModel.fromFirestore(doc.data(), doc.id))
            .toList();
      }
    } catch (_) {
      final fallback = await _fallbackMedicines(category: category);
      yield fallback;
    }
  }

  Future<MedicineModel?> getMedicineById(String id) async {
    try {
      final doc = await _db.collection('medicines').doc(id).get();
      if (!doc.exists) return null;
      return MedicineModel.fromFirestore(
        doc.data() as Map<String, dynamic>,
        doc.id,
      );
    } catch (_) {
      final fallback = await LocalDataService.getMedicineById(id);
      if (fallback == null) return null;
      return MedicineModel.fromFirestore(fallback, fallback['id']?.toString() ?? id);
    }
  }

  Future<List<MedicineModel>> searchMedicines(String query) async {
    if (query.isEmpty) return [];

    try {
      final snapshot = await _db
          .collection('medicines')
          .where('isActive', isEqualTo: true)
          .orderBy('name')
          .startAt([query])
          .endAt(['$query\uf8ff'])
          .limit(20)
          .get();

      return snapshot.docs
          .map((doc) => MedicineModel.fromFirestore(doc.data(), doc.id))
          .toList();
    } catch (_) {
      final fallback = await LocalDataService.searchMedicines(query);
      return fallback
          .map((item) => MedicineModel.fromFirestore(
                Map<String, dynamic>.from(item as Map),
                (item as Map)['id']?.toString() ?? '',
              ))
          .toList();
    }
  }

  Future<List<MedicineModel>> _fallbackMedicines({String? category}) async {
    final fallback = await LocalDataService.getMedicines(category: category);
    return fallback
        .map((item) => MedicineModel.fromFirestore(
              Map<String, dynamic>.from(item as Map),
              (item as Map)['id']?.toString() ?? '',
            ))
        .toList();
  }

  Future<String> placeOrder(OrderModel order) async {
    await _rateLimit.checkOrderRateLimit(order.userId);

    final sanitizedName = _sanitizeString(order.userName);
    final sanitizedNotes = order.notes != null ? _sanitizeString(order.notes!) : null;

    final orderId = order.orderId.isNotEmpty ? order.orderId : placeOrderId();
    final orderWithId = OrderModel(
      orderId: orderId,
      userId: order.userId,
      userPhone: order.userPhone,
      userName: sanitizedName,
      deliveryAddress: order.deliveryAddress,
      items: order.items,
      prescriptionUrl: order.prescriptionUrl,
      totalAmount: order.totalAmount,
      status: order.status,
      paymentMethod: order.paymentMethod,
      paymentStatus: order.paymentStatus,
      notes: sanitizedNotes,
      createdAt: order.createdAt,
      updatedAt: order.updatedAt,
    );

    await _db.collection('orders').doc(orderId).set(orderWithId.toMap());
    
    AppLogger.info("Order placed: $orderId", tag: "Order");
    AppLogger.logCustomKey("order_id", orderId);
    AppLogger.logCustomKey("order_total", order.totalAmount.toString());
    
    return orderId;
  }

  String _sanitizeString(String input) {
    return input
        .trim()
        .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), '')
        .replaceAll(RegExp(r'<[^>]*>'), '');
  }

  Stream<List<OrderModel>> getUserOrders(String userId) {
    return _db
        .collection('orders')
        .where('userId', isEqualTo: userId)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snapshot) {
      return snapshot.docs
          .map((doc) => OrderModel.fromFirestore(doc.data(), doc.id))
          .toList();
    });
  }

  Stream<List<OrderModel>> getAllOrders() {
    return _db
        .collection('orders')
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snapshot) {
      return snapshot.docs
          .map((doc) => OrderModel.fromFirestore(doc.data(), doc.id))
          .toList();
    });
  }

  Future<({List<OrderModel> orders, DocumentSnapshot? lastDoc})> getOrdersPaginated({
    DocumentSnapshot? lastDoc,
    int limit = 20,
  }) async {
    Query<Map<String, dynamic>> query = _db
        .collection('orders')
        .orderBy('createdAt', descending: true)
        .limit(limit);

    if (lastDoc != null) {
      query = query.startAfterDocument(lastDoc);
    }

    final snapshot = await query.get();
    return (
      orders: snapshot.docs
          .map((doc) => OrderModel.fromFirestore(doc.data(), doc.id))
          .toList(),
      lastDoc: snapshot.docs.isNotEmpty ? snapshot.docs.last : null,
    );
  }

  Stream<List<OrderModel>> getOrdersByStatus(String status) {
    return _db
        .collection('orders')
        .where('status', isEqualTo: status)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snapshot) {
      return snapshot.docs
          .map((doc) => OrderModel.fromFirestore(doc.data(), doc.id))
          .toList();
    });
  }

  Future<OrderModel?> getOrderById(String orderId) async {
    final doc = await _db.collection('orders').doc(orderId).get();
    if (doc.exists) {
      return OrderModel.fromFirestore(doc.data()!, doc.id);
    }
    return null;
  }

  Future<void> updateOrderStatus(String orderId, String status) async {
    await _db.collection('orders').doc(orderId).update({
      'status': status,
      'updatedAt': Timestamp.fromDate(DateTime.now()),
    });
  }

  Future<UserModel?> getUser(String uid) async {
    final doc = await _db.collection('users').doc(uid).get();
    if (doc.exists) {
      return UserModel.fromFirestore(doc.data()!, uid);
    }
    return null;
  }

  Stream<UserModel?> getUserStream(String uid) {
    return _db.collection('users').doc(uid).snapshots().map((doc) {
      if (doc.exists) {
        return UserModel.fromFirestore(doc.data()!, doc.id);
      }
      return null;
    });
  }

  Future<void> updateUser(String uid, Map<String, dynamic> data) async {
    await _db.collection('users').doc(uid).update(data);
  }

  Stream<List<CategoryModel>> getCategoriesStream() {
    return _db.collection('categories').orderBy('order').snapshots().map((snapshot) {
      return snapshot.docs
          .map((doc) => CategoryModel.fromFirestore(doc.data(), doc.id))
          .toList();
    });
  }

  Future<void> addMedicine(MedicineModel medicine) async {
    final id = _uuid.v4();
    await _db.collection('medicines').doc(id).set(
          MedicineModel(
            id: id,
            name: medicine.name,
            genericName: medicine.genericName,
            manufacturer: medicine.manufacturer,
            category: medicine.category,
            price: medicine.price,
            mrp: medicine.mrp,
            stock: medicine.stock,
            unit: medicine.unit,
            imageUrl: medicine.imageUrl,
            requiresPrescription: medicine.requiresPrescription,
            description: medicine.description,
            isActive: medicine.isActive,
          ).toMap(),
        );
  }
}
