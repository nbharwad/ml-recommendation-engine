import 'package:cloud_firestore/cloud_firestore.dart';

class UserModel {
  final String uid;
  final String name;
  final String phone;
  final String address;
  final String pincode;
  final bool isAdmin;
  final String? fcmToken;
  final DateTime createdAt;

  UserModel({
    required this.uid,
    required this.name,
    required this.phone,
    this.address = '',
    this.pincode = '',
    this.isAdmin = false,
    this.fcmToken,
    required this.createdAt,
  });

  factory UserModel.fromFirestore(Map<String, dynamic> data, String uid) {
    return UserModel(
      uid: uid,
      name: data['name'] ?? '',
      phone: data['phone'] ?? '',
      address: data['address'] ?? '',
      pincode: data['pincode'] ?? '',
      isAdmin: data['isAdmin'] ?? false,
      fcmToken: data['fcmToken'],
      createdAt: (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'name': name,
      'phone': phone,
      'address': address,
      'pincode': pincode,
      'fcmToken': fcmToken,
      'createdAt': Timestamp.fromDate(createdAt),
    };
  }

  UserModel copyWith({
    String? uid,
    String? name,
    String? phone,
    String? address,
    String? pincode,
    bool? isAdmin,
    String? fcmToken,
    DateTime? createdAt,
  }) {
    return UserModel(
      uid: uid ?? this.uid,
      name: name ?? this.name,
      phone: phone ?? this.phone,
      address: address ?? this.address,
      pincode: pincode ?? this.pincode,
      isAdmin: isAdmin ?? this.isAdmin,
      fcmToken: fcmToken ?? this.fcmToken,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}
