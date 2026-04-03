class MedicineModel {
  final String id;
  final String name;
  final String genericName;
  final String manufacturer;
  final String category;
  final double price;
  final double mrp;
  final int stock;
  final String unit;
  final String imageUrl;
  final bool requiresPrescription;
  final String description;
  final bool isActive;

  MedicineModel({
    required this.id,
    required this.name,
    this.genericName = '',
    this.manufacturer = '',
    required this.category,
    required this.price,
    required this.mrp,
    required this.stock,
    this.unit = 'strip',
    this.imageUrl = '',
    this.requiresPrescription = false,
    this.description = '',
    this.isActive = true,
  });

  factory MedicineModel.fromFirestore(Map<String, dynamic> data, String id) {
    double parseDouble(dynamic value) {
      if (value is num) return value.toDouble();
      return double.tryParse(value?.toString() ?? '') ?? 0;
    }

    int parseInt(dynamic value, {int fallback = 0}) {
      if (value is int) return value;
      if (value is num) return value.toInt();
      return int.tryParse(value?.toString() ?? '') ?? fallback;
    }

    bool parseBool(dynamic value) {
      if (value is bool) return value;
      final normalized = value?.toString().trim().toLowerCase();
      return normalized == 'true' || normalized == '1' || normalized == 'yes';
    }

    return MedicineModel(
      id: id,
      name: data['name'] ?? '',
      genericName: data['composition'] ?? data['genericName'] ?? '',
      manufacturer: data['manufacturer'] ?? '',
      category: data['category'] ?? 'other',
      price: parseDouble(data['price']),
      mrp: parseDouble(data['mrp'] ?? data['price']),
      stock: parseInt(data['stock'], fallback: 50),
      unit: data['unit'] ?? 'strip',
      imageUrl: data['imageUrl'] ?? '',
      requiresPrescription: parseBool(data['requiresPrescription']),
      description: data['composition'] ?? data['description'] ?? '',
      isActive: data['isActive'] == null ? true : parseBool(data['isActive']),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'genericName': genericName,
      'manufacturer': manufacturer,
      'category': category,
      'price': price,
      'mrp': mrp,
      'stock': stock,
      'unit': unit,
      'imageUrl': imageUrl,
      'requiresPrescription': requiresPrescription,
      'description': description,
      'isActive': isActive,
    };
  }

  double get discountPercentage {
    if (mrp <= 0 || price >= mrp) return 0;
    return ((mrp - price) / mrp * 100).roundToDouble();
  }

  bool get isInStock => stock > 0;
}
