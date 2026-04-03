import 'medicine_model.dart';

class CartItem {
  final MedicineModel medicine;
  int quantity;

  CartItem({required this.medicine, this.quantity = 1});

  double get subtotal => medicine.price * quantity;

  CartItem copyWith({
    MedicineModel? medicine,
    int? quantity,
  }) {
    return CartItem(
      medicine: medicine ?? this.medicine,
      quantity: quantity ?? this.quantity,
    );
  }
}
