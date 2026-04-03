class CategoryModel {
  final String id;
  final String name;
  final String icon;
  final int order;

  CategoryModel({
    required this.id,
    required this.name,
    required this.icon,
    this.order = 0,
  });

  factory CategoryModel.fromFirestore(Map<String, dynamic> data, String id) {
    return CategoryModel(
      id: id,
      name: data['name'] ?? '',
      icon: data['icon'] ?? '',
      order: data['order'] ?? 0,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'icon': icon,
      'order': order,
    };
  }

  static List<CategoryModel> get defaultCategories => [
        CategoryModel(id: 'all', name: 'All', icon: 'All', order: 0),
        CategoryModel(id: 'pain_relief', name: 'Pain Relief', icon: 'PR', order: 1),
        CategoryModel(id: 'fever', name: 'Fever', icon: 'FV', order: 2),
        CategoryModel(id: 'vitamins', name: 'Vitamins', icon: 'VT', order: 3),
        CategoryModel(id: 'antibiotics', name: 'Antibiotics', icon: 'AB', order: 4),
        CategoryModel(id: 'other', name: 'Other', icon: 'OT', order: 5),
      ];
}
