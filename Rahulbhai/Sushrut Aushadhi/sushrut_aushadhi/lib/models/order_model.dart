import 'package:cloud_firestore/cloud_firestore.dart';

enum OrderStatus {
  pending,
  confirmed,
  preparing,
  outForDelivery,
  delivered,
  cancelled;

  String get displayName {
    switch (this) {
      case OrderStatus.pending:
        return 'Pending';
      case OrderStatus.confirmed:
        return 'Confirmed';
      case OrderStatus.preparing:
        return 'Preparing';
      case OrderStatus.outForDelivery:
        return 'Out for Delivery';
      case OrderStatus.delivered:
        return 'Delivered';
      case OrderStatus.cancelled:
        return 'Cancelled';
    }
  }

  static OrderStatus fromString(String status) {
    return OrderStatus.values.firstWhere(
      (e) => e.name == status.toLowerCase(),
      orElse: () => OrderStatus.pending,
    );
  }
}

class OrderItem {
  final String medicineId;
  final String medicineName;
  final double price;
  final int quantity;
  final double subtotal;

  OrderItem({
    required this.medicineId,
    required this.medicineName,
    required this.price,
    required this.quantity,
    required this.subtotal,
  });

  factory OrderItem.fromMap(Map<String, dynamic> data) {
    return OrderItem(
      medicineId: data['medicineId'] ?? '',
      medicineName: data['medicineName'] ?? '',
      price: (data['price'] ?? 0).toDouble(),
      quantity: data['quantity'] ?? 1,
      subtotal: (data['subtotal'] ?? 0).toDouble(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'medicineId': medicineId,
      'medicineName': medicineName,
      'price': price,
      'quantity': quantity,
      'subtotal': subtotal,
    };
  }
}

class StatusHistoryEntry {
  final String status;
  final DateTime timestamp;
  final String updatedBy;
  final String? role;
  final String? note;

  StatusHistoryEntry({
    required this.status,
    required this.timestamp,
    required this.updatedBy,
    this.role,
    this.note,
  });

  factory StatusHistoryEntry.fromMap(Map<String, dynamic> data) {
    return StatusHistoryEntry(
      status: data['status'] ?? '',
      timestamp: (data['timestamp'] as Timestamp?)?.toDate() ?? DateTime.now(),
      updatedBy: data['updatedBy'] ?? 'system',
      role: data['role'] as String?,
      note: data['note'] as String?,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'status': status,
      'timestamp': Timestamp.fromDate(timestamp),
      'updatedBy': updatedBy,
      if (role != null) 'role': role,
      if (note != null) 'note': note,
    };
  }
}

class OrderModel {
  final String orderId;
  final String userId;
  final String userPhone;
  final String userName;
  final String deliveryAddress;
  final List<OrderItem> items;
  final String? prescriptionUrl;
  final double totalAmount;
  final OrderStatus status;
  final String paymentMethod;
  final String paymentStatus;
  final String? notes;
  final int? rating;
  final DateTime createdAt;
  final DateTime updatedAt;
  final List<StatusHistoryEntry> statusHistory;

  OrderModel({
    required this.orderId,
    required this.userId,
    required this.userPhone,
    required this.userName,
    required this.deliveryAddress,
    required this.items,
    this.prescriptionUrl,
    required this.totalAmount,
    this.status = OrderStatus.pending,
    this.paymentMethod = 'cod',
    this.paymentStatus = 'pending',
    this.notes,
    this.rating,
    required this.createdAt,
    required this.updatedAt,
    this.statusHistory = const [],
  });

  factory OrderModel.fromFirestore(Map<String, dynamic> data, String id) {
    return OrderModel(
      orderId: id,
      userId: data['userId'] ?? '',
      userPhone: data['userPhone'] ?? '',
      userName: data['userName'] ?? '',
      deliveryAddress: data['deliveryAddress'] ?? '',
      items: (data['items'] as List?)
              ?.map((e) => OrderItem.fromMap(e as Map<String, dynamic>))
              .toList() ??
          [],
      prescriptionUrl: data['prescriptionUrl'],
      totalAmount: (data['totalAmount'] ?? 0).toDouble(),
      status: OrderStatus.fromString(data['status'] ?? 'pending'),
      paymentMethod: data['paymentMethod'] ?? 'cod',
      paymentStatus: data['paymentStatus'] ?? 'pending',
      notes: data['notes'],
      rating: data['rating'],
      createdAt: (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
      updatedAt: (data['updatedAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
      statusHistory: (data['statusHistory'] as List?)
              ?.map((e) => StatusHistoryEntry.fromMap(e as Map<String, dynamic>))
              .toList() ??
          [],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'orderId': orderId,
      'userId': userId,
      'userPhone': userPhone,
      'userName': userName,
      'deliveryAddress': deliveryAddress,
      'items': items.map((e) => e.toMap()).toList(),
      'prescriptionUrl': prescriptionUrl,
      'totalAmount': totalAmount,
      'status': status.name,
      'paymentMethod': paymentMethod,
      'paymentStatus': paymentStatus,
      'notes': notes,
      'rating': rating,
      'createdAt': Timestamp.fromDate(createdAt),
      'updatedAt': Timestamp.fromDate(updatedAt),
      'statusHistory': statusHistory.map((e) => e.toMap()).toList(),
    };
  }

  OrderModel copyWith({
    String? orderId,
    String? userId,
    String? userPhone,
    String? userName,
    String? deliveryAddress,
    List<OrderItem>? items,
    String? prescriptionUrl,
    double? totalAmount,
    OrderStatus? status,
    String? paymentMethod,
    String? paymentStatus,
    String? notes,
    int? rating,
    DateTime? createdAt,
    DateTime? updatedAt,
    List<StatusHistoryEntry>? statusHistory,
  }) {
    return OrderModel(
      orderId: orderId ?? this.orderId,
      userId: userId ?? this.userId,
      userPhone: userPhone ?? this.userPhone,
      userName: userName ?? this.userName,
      deliveryAddress: deliveryAddress ?? this.deliveryAddress,
      items: items ?? this.items,
      prescriptionUrl: prescriptionUrl ?? this.prescriptionUrl,
      totalAmount: totalAmount ?? this.totalAmount,
      status: status ?? this.status,
      paymentMethod: paymentMethod ?? this.paymentMethod,
      paymentStatus: paymentStatus ?? this.paymentStatus,
      notes: notes ?? this.notes,
      rating: rating ?? this.rating,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      statusHistory: statusHistory ?? this.statusHistory,
    );
  }

  int get itemCount => items.fold(0, (sum, item) => sum + item.quantity);
}
