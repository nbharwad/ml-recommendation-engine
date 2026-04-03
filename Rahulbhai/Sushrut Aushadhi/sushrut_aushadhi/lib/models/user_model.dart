import 'package:cloud_firestore/cloud_firestore.dart';

import 'delivery_address.dart';

class UserModel {
  final String uid;
  final String name;
  final String phone;
  final String email;
  final String address;
  final String pincode;
  final DeliveryAddress deliveryAddress;
  final bool isAdmin;
  final String? fcmToken;
  final DateTime createdAt;

  UserModel({
    required this.uid,
    required this.name,
    required this.phone,
    this.email = '',
    this.address = '',
    this.pincode = '',
    DeliveryAddress? deliveryAddress,
    this.isAdmin = false,
    this.fcmToken,
    required this.createdAt,
  }) : deliveryAddress =
            deliveryAddress ?? DeliveryAddress(line1: '', city: '', state: '', pincode: '');

  factory UserModel.fromFirestore(Map<String, dynamic> data, String uid) {
    return UserModel(
      uid: uid,
      name: data['name'] ?? '',
      phone: data['phone'] ?? '',
      email: data['email'] ?? '',
      address: data['address'] ?? '',
      pincode: data['pincode'] ?? '',
      deliveryAddress: DeliveryAddress.fromFirestoreData(data['deliveryAddress']),
      isAdmin: data['isAdmin'] ?? false,
      fcmToken: data['fcmToken'],
      createdAt: (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'name': name,
      'phone': phone,
      'email': email,
      'address': address,
      'pincode': pincode,
      'deliveryAddress': deliveryAddress.toMap(),
      'fcmToken': fcmToken,
      'createdAt': Timestamp.fromDate(createdAt),
    };
  }

  UserModel copyWith({
    String? uid,
    String? name,
    String? phone,
    String? email,
    String? address,
    String? pincode,
    DeliveryAddress? deliveryAddress,
    bool? isAdmin,
    String? fcmToken,
    DateTime? createdAt,
  }) {
    return UserModel(
      uid: uid ?? this.uid,
      name: name ?? this.name,
      phone: phone ?? this.phone,
      email: email ?? this.email,
      address: address ?? this.address,
      pincode: pincode ?? this.pincode,
      deliveryAddress: deliveryAddress ?? this.deliveryAddress,
      isAdmin: isAdmin ?? this.isAdmin,
      fcmToken: fcmToken ?? this.fcmToken,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}
