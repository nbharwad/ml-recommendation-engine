import 'package:cloud_firestore/cloud_firestore.dart';

enum LabOrderStatus {
  pending,
  sampleCollected,
  processing,
  completed,
  cancelled;

  String get displayName {
    switch (this) {
      case LabOrderStatus.pending:
        return 'Pending';
      case LabOrderStatus.sampleCollected:
        return 'Sample Collected';
      case LabOrderStatus.processing:
        return 'Processing';
      case LabOrderStatus.completed:
        return 'Completed';
      case LabOrderStatus.cancelled:
        return 'Cancelled';
    }
  }

  static LabOrderStatus fromString(String status) {
    return LabOrderStatus.values.firstWhere(
      (e) => e.name == status.toLowerCase(),
      orElse: () => LabOrderStatus.pending,
    );
  }
}

class LabTestItem {
  final String testId;
  final String testName;
  final double price;

  LabTestItem({
    required this.testId,
    required this.testName,
    required this.price,
  });

  factory LabTestItem.fromMap(Map<String, dynamic> data) {
    return LabTestItem(
      testId: data['testId'] ?? '',
      testName: data['testName'] ?? '',
      price: (data['price'] ?? 0).toDouble(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'testId': testId,
      'testName': testName,
      'price': price,
    };
  }
}

class LabTestModel {
  final String id;
  final String name;
  final String category;
  final double price;
  final int tatHours;
  final String sampleType;
  final bool active;

  LabTestModel({
    required this.id,
    required this.name,
    required this.category,
    required this.price,
    required this.tatHours,
    required this.sampleType,
    this.active = true,
  });

  factory LabTestModel.fromFirestore(Map<String, dynamic> data, String id) {
    // Safe int conversion helper
    int toInt(dynamic value) {
      if (value == null) return 24;
      if (value is int) return value;
      if (value is double) return value.toInt();
      if (value is String) return int.tryParse(value) ?? 24;
      return 24;
    }

    // Safe double conversion helper
    double toDouble(dynamic value) {
      if (value == null) return 0.0;
      if (value is double) return value;
      if (value is int) return value.toDouble();
      if (value is String) return double.tryParse(value) ?? 0.0;
      return 0.0;
    }

    return LabTestModel(
      id: id,
      name: data['name']?.toString() ?? '',
      category: data['category']?.toString() ?? '',
      price: toDouble(data['price']),
      tatHours: toInt(data['tat_hours']),
      sampleType: data['sample_type']?.toString() ?? '',
      active: data['active'] ?? true,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'name': name,
      'category': category,
      'price': price,
      'tat_hours': tatHours,
      'sample_type': sampleType,
      'active': active,
    };
  }
}

class LabStatusHistoryEntry {
  final String status;
  final DateTime timestamp;
  final String? updatedBy;
  final String? note;

  LabStatusHistoryEntry({
    required this.status,
    required this.timestamp,
    this.updatedBy,
    this.note,
  });

  factory LabStatusHistoryEntry.fromMap(Map<String, dynamic> data) {
    return LabStatusHistoryEntry(
      status: data['status'] ?? '',
      timestamp: (data['timestamp'] as Timestamp?)?.toDate() ?? DateTime.now(),
      updatedBy: data['updatedBy'],
      note: data['note'],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'status': status,
      'timestamp': Timestamp.fromDate(timestamp),
      if (updatedBy != null) 'updatedBy': updatedBy,
      if (note != null) 'note': note,
    };
  }
}

class LabOrderModel {
  final String orderId;
  final String userId;
  final String userPhone;
  final String userName;
  final String? homeCollectionAddress;
  final List<LabTestItem> tests;
  final double totalAmount;
  final LabOrderStatus status;
  final String paymentMethod;
  final String paymentStatus;
  final DateTime? paidAt;
  final bool resultUploaded;
  final String? notes;
  final DateTime createdAt;
  final DateTime updatedAt;
  final DateTime? completedAt;
  final List<LabStatusHistoryEntry> statusHistory;
  final String? labResultUrl;

  LabOrderModel({
    required this.orderId,
    required this.userId,
    required this.userPhone,
    required this.userName,
    this.homeCollectionAddress,
    required this.tests,
    required this.totalAmount,
    this.status = LabOrderStatus.pending,
    this.paymentMethod = 'cod',
    this.paymentStatus = 'pending',
    this.paidAt,
    this.resultUploaded = false,
    this.notes,
    required this.createdAt,
    required this.updatedAt,
    this.completedAt,
    this.statusHistory = const [],
    this.labResultUrl,
  });

  int get testCount => tests.length;

  factory LabOrderModel.fromFirestore(Map<String, dynamic> data, String id) {
    return LabOrderModel(
      orderId: id,
      userId: data['userId'] ?? '',
      userPhone: data['userPhone'] ?? '',
      userName: data['userName'] ?? '',
      homeCollectionAddress: data['homeCollectionAddress'],
      tests: (data['tests'] as List?)
              ?.map((e) => LabTestItem.fromMap(e as Map<String, dynamic>))
              .toList() ??
          [],
      totalAmount: (data['totalAmount'] ?? 0).toDouble(),
      status: LabOrderStatus.fromString(data['status'] ?? 'pending'),
      paymentMethod: data['paymentMethod'] ?? 'cod',
      paymentStatus: data['paymentStatus'] ?? 'pending',
      paidAt: (data['paidAt'] as Timestamp?)?.toDate(),
      resultUploaded: data['resultUploaded'] ?? false,
      notes: data['notes'],
      createdAt: (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
      updatedAt: (data['updatedAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
      completedAt: (data['completedAt'] as Timestamp?)?.toDate(),
      statusHistory: (data['statusHistory'] as List?)
              ?.map((e) => LabStatusHistoryEntry.fromMap(e as Map<String, dynamic>))
              .toList() ??
          [],
      labResultUrl: data['labResultUrl'],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'orderId': orderId,
      'userId': userId,
      'userPhone': userPhone,
      'userName': userName,
      if (homeCollectionAddress != null) 'homeCollectionAddress': homeCollectionAddress,
      'tests': tests.map((e) => e.toMap()).toList(),
      'totalAmount': totalAmount,
      'status': status.name,
      'paymentMethod': paymentMethod,
      'paymentStatus': paymentStatus,
      if (paidAt != null) 'paidAt': Timestamp.fromDate(paidAt!),
      'resultUploaded': resultUploaded,
      if (notes != null) 'notes': notes,
      'createdAt': Timestamp.fromDate(createdAt),
      'updatedAt': Timestamp.fromDate(updatedAt),
      if (completedAt != null) 'completedAt': Timestamp.fromDate(completedAt!),
      'statusHistory': statusHistory.map((e) => e.toMap()).toList(),
      if (labResultUrl != null) 'labResultUrl': labResultUrl,
    };
  }

  LabOrderModel copyWith({
    String? orderId,
    String? userId,
    String? userPhone,
    String? userName,
    String? homeCollectionAddress,
    List<LabTestItem>? tests,
    double? totalAmount,
    LabOrderStatus? status,
    String? paymentMethod,
    String? paymentStatus,
    DateTime? paidAt,
    bool? resultUploaded,
    String? notes,
    DateTime? createdAt,
    DateTime? updatedAt,
    DateTime? completedAt,
    List<LabStatusHistoryEntry>? statusHistory,
    String? labResultUrl,
  }) {
    return LabOrderModel(
      orderId: orderId ?? this.orderId,
      userId: userId ?? this.userId,
      userPhone: userPhone ?? this.userPhone,
      userName: userName ?? this.userName,
      homeCollectionAddress: homeCollectionAddress ?? this.homeCollectionAddress,
      tests: tests ?? this.tests,
      totalAmount: totalAmount ?? this.totalAmount,
      status: status ?? this.status,
      paymentMethod: paymentMethod ?? this.paymentMethod,
      paymentStatus: paymentStatus ?? this.paymentStatus,
      paidAt: paidAt ?? this.paidAt,
      resultUploaded: resultUploaded ?? this.resultUploaded,
      notes: notes ?? this.notes,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      completedAt: completedAt ?? this.completedAt,
      statusHistory: statusHistory ?? this.statusHistory,
      labResultUrl: labResultUrl ?? this.labResultUrl,
    );
  }
}