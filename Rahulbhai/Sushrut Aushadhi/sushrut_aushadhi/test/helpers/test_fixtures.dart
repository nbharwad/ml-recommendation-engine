import 'package:sushrut_aushadhi/models/medicine_model.dart';
import 'package:sushrut_aushadhi/models/order_model.dart';
import 'package:sushrut_aushadhi/models/user_model.dart';
import 'package:sushrut_aushadhi/models/cart_item_model.dart';
import 'package:sushrut_aushadhi/models/delivery_address.dart';

class TestFixtures {
  static MedicineModel createMedicine({
    String? id,
    String? name,
    double? price,
    double? mrp,
    int? stock,
    String? category,
  }) {
    return MedicineModel(
      id: id ?? 'med-001',
      name: name ?? 'Test Medicine',
      genericName: 'Test Generic',
      manufacturer: 'Test Manufacturer',
      category: category ?? 'General',
      price: price ?? 99.99,
      mrp: mrp ?? 149.99,
      stock: stock ?? 100,
      unit: 'strip',
      imageUrl: 'https://example.com/image.jpg',
      requiresPrescription: false,
      description: 'Test description',
      isActive: true,
    );
  }

  static UserModel createUser({
    String? uid,
    String? name,
    String? phone,
    bool? isAdmin,
  }) {
    return UserModel(
      uid: uid ?? 'user-123',
      name: name ?? 'Test User',
      phone: phone ?? '+911234567890',
      deliveryAddress: DeliveryAddress(
        line1: '123 Test Street',
        city: 'Test City',
        state: 'Test State',
        pincode: '123456',
      ),
      isAdmin: isAdmin ?? false,
      createdAt: DateTime.now(),
    );
  }

  static OrderModel createOrder({
    String? orderId,
    String? userId,
    OrderStatus? status,
    List<OrderItem>? items,
    double? totalAmount,
  }) {
    return OrderModel(
      orderId: orderId ?? 'order-001',
      userId: userId ?? 'user-123',
      userPhone: '+911234567890',
      userName: 'Test User',
      deliveryAddress: DeliveryAddress(
        line1: '123 Test Street',
        city: 'Test City',
        state: 'Test State',
        pincode: '123456',
      ),
      items: items ?? [createOrderItem()],
      totalAmount: totalAmount ?? 199.98,
      status: status ?? OrderStatus.pending,
      paymentMethod: 'cod',
      paymentStatus: 'pending',
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
    );
  }

  static OrderItem createOrderItem({
    String? medicineId,
    String? medicineName,
    double? price,
    int? quantity,
  }) {
    final itemPrice = price ?? 99.99;
    final itemQuantity = quantity ?? 2;
    return OrderItem(
      medicineId: medicineId ?? 'med-001',
      medicineName: medicineName ?? 'Test Medicine',
      price: itemPrice,
      quantity: itemQuantity,
      subtotal: itemPrice * itemQuantity,
    );
  }

  static CartItem createCartItem({
    MedicineModel? medicine,
    int? quantity,
  }) {
    return CartItem(
      medicine: medicine ?? createMedicine(),
      quantity: quantity ?? 2,
    );
  }

  static Map<String, dynamic> createMedicineMap({
    String? id,
    String? name,
    double? price,
  }) {
    return {
      'id': id ?? 'med-001',
      'name': name ?? 'Test Medicine',
      'composition': 'Test Composition',
      'manufacturer': 'Test Manufacturer',
      'category': 'General',
      'price': price ?? 99.99,
      'mrp': 149.99,
      'stock': 100,
      'unit': 'strip',
      'imageUrl': '',
      'requiresPrescription': false,
      'description': 'Test description',
      'isActive': true,
    };
  }
}
