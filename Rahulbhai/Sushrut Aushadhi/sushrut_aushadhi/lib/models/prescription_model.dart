import 'package:cloud_firestore/cloud_firestore.dart';

enum PrescriptionStatus {
  pending,
  approved,
  rejected;

  String get displayName {
    switch (this) {
      case PrescriptionStatus.pending:
        return 'Pending';
      case PrescriptionStatus.approved:
        return 'Approved';
      case PrescriptionStatus.rejected:
        return 'Rejected';
    }
  }

  static PrescriptionStatus fromString(String? value) {
    switch (value) {
      case 'approved':
        return PrescriptionStatus.approved;
      case 'rejected':
        return PrescriptionStatus.rejected;
      default:
        return PrescriptionStatus.pending;
    }
  }
}

class PrescriptionModel {
  final String id;
  final String userId;
  final String? userName;
  final String? userPhone;
  final String imageUrl;
  final PrescriptionStatus status;
  final String? notes;
  final DateTime createdAt;
  final DateTime updatedAt;

  PrescriptionModel({
    required this.id,
    required this.userId,
    this.userName,
    this.userPhone,
    required this.imageUrl,
    this.status = PrescriptionStatus.pending,
    this.notes,
    required this.createdAt,
    required this.updatedAt,
  });

  factory PrescriptionModel.fromFirestore(Map<String, dynamic> data, String id) {
    return PrescriptionModel(
      id: id,
      userId: data['userId'] ?? '',
      userName: data['userName'],
      userPhone: data['userPhone'],
      imageUrl: data['imageUrl'] ?? '',
      status: PrescriptionStatus.fromString(data['status']),
      notes: data['notes'],
      createdAt: (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
      updatedAt: (data['updatedAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'userId': userId,
      'userName': userName,
      'userPhone': userPhone,
      'imageUrl': imageUrl,
      'status': status.name,
      'notes': notes,
      'createdAt': Timestamp.fromDate(createdAt),
      'updatedAt': Timestamp.fromDate(updatedAt),
    };
  }

  PrescriptionModel copyWith({
    String? id,
    String? userId,
    String? userName,
    String? userPhone,
    String? imageUrl,
    PrescriptionStatus? status,
    String? notes,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return PrescriptionModel(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      userName: userName ?? this.userName,
      userPhone: userPhone ?? this.userPhone,
      imageUrl: imageUrl ?? this.imageUrl,
      status: status ?? this.status,
      notes: notes ?? this.notes,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }
}