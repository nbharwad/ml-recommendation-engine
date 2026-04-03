import 'package:flutter_test/flutter_test.dart';
import 'package:sushrut_aushadhi/models/medicine_model.dart';

void main() {
  group('MedicineModel', () {
    group('fromFirestore', () {
      test('should create MedicineModel from valid Firestore data', () {
        final data = {
          'name': 'Paracetamol',
          'composition': 'Acetaminophen',
          'manufacturer': 'Cipla',
          'category': 'Pain Relief',
          'price': 30.0,
          'mrp': 45.0,
          'stock': 100,
          'unit': 'strip',
          'imageUrl': 'https://example.com/paracetamol.jpg',
          'requiresPrescription': false,
          'description': 'Fever and pain reliever',
          'isActive': true,
        };

        final medicine = MedicineModel.fromFirestore(data, 'med-001');

        expect(medicine.id, equals('med-001'));
        expect(medicine.name, equals('Paracetamol'));
        expect(medicine.genericName, equals('Acetaminophen'));
        expect(medicine.manufacturer, equals('Cipla'));
        expect(medicine.category, equals('Pain Relief'));
        expect(medicine.price, equals(30.0));
        expect(medicine.mrp, equals(45.0));
        expect(medicine.stock, equals(100));
        expect(medicine.unit, equals('strip'));
        expect(medicine.requiresPrescription, isFalse);
        expect(medicine.isActive, isTrue);
      });

      test('should handle numeric price values', () {
        final data = {
          'name': 'Medicine',
          'category': 'General',
          'price': 100,
          'mrp': 150,
          'stock': 50,
        };

        final medicine = MedicineModel.fromFirestore(data, 'med-001');

        expect(medicine.price, equals(100.0));
        expect(medicine.mrp, equals(150.0));
      });

      test('should handle string price values', () {
        final data = {
          'name': 'Medicine',
          'category': 'General',
          'price': '99.50',
          'mrp': '149.50',
          'stock': '50',
        };

        final medicine = MedicineModel.fromFirestore(data, 'med-001');

        expect(medicine.price, equals(99.50));
        expect(medicine.mrp, equals(149.50));
      });

      test('should handle missing optional fields with defaults', () {
        final data = {
          'name': 'Medicine',
          'category': 'General',
          'price': 100,
          'mrp': 150,
        };

        final medicine = MedicineModel.fromFirestore(data, 'med-001');

        expect(medicine.genericName, equals(''));
        expect(medicine.manufacturer, equals(''));
        expect(medicine.unit, equals('strip'));
        expect(medicine.imageUrl, equals(''));
        expect(medicine.requiresPrescription, isFalse);
        expect(medicine.isActive, isTrue);
        expect(medicine.stock, equals(50)); // fallback
      });

      test('should parse boolean from various string values', () {
        expect(MedicineModel.fromFirestore(
          {'name': 'M1', 'category': 'C1', 'price': 10, 'mrp': 10, 'requiresPrescription': 'true'},
          'id',
        ).requiresPrescription, isTrue);

        expect(MedicineModel.fromFirestore(
          {'name': 'M1', 'category': 'C1', 'price': 10, 'mrp': 10, 'requiresPrescription': '1'},
          'id',
        ).requiresPrescription, isTrue);

        expect(MedicineModel.fromFirestore(
          {'name': 'M1', 'category': 'C1', 'price': 10, 'mrp': 10, 'requiresPrescription': 'yes'},
          'id',
        ).requiresPrescription, isTrue);

        expect(MedicineModel.fromFirestore(
          {'name': 'M1', 'category': 'C1', 'price': 10, 'mrp': 10, 'requiresPrescription': 'false'},
          'id',
        ).requiresPrescription, isFalse);
      });
    });

    group('toMap', () {
      test('should convert MedicineModel to Map', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Paracetamol',
          genericName: 'Acetaminophen',
          manufacturer: 'Cipla',
          category: 'Pain Relief',
          price: 30.0,
          mrp: 45.0,
          stock: 100,
          unit: 'strip',
          imageUrl: 'https://example.com/image.jpg',
          requiresPrescription: false,
          description: 'Fever reducer',
          isActive: true,
        );

        final map = medicine.toMap();

        expect(map['id'], equals('med-001'));
        expect(map['name'], equals('Paracetamol'));
        expect(map['genericName'], equals('Acetaminophen'));
        expect(map['manufacturer'], equals('Cipla'));
        expect(map['category'], equals('Pain Relief'));
        expect(map['price'], equals(30.0));
        expect(map['mrp'], equals(45.0));
        expect(map['stock'], equals(100));
        expect(map['unit'], equals('strip'));
        expect(map['requiresPrescription'], isFalse);
        expect(map['isActive'], isTrue);
      });

      test('should be reversible (fromFirestore -> toMap)', () {
        final originalData = {
          'name': 'Test Medicine',
          'composition': 'Test Composition',
          'manufacturer': 'Test Mfg',
          'category': 'Test Category',
          'price': 100.0,
          'mrp': 150.0,
          'stock': 75,
          'unit': 'tablet',
          'imageUrl': 'http://test.com/image.jpg',
          'requiresPrescription': true,
          'description': 'Test description',
          'isActive': false,
        };

        final medicine = MedicineModel.fromFirestore(originalData, 'test-id');
        final map = medicine.toMap();

        expect(map['name'], equals(originalData['name']));
        expect(map['price'], equals(originalData['price']));
        expect(map['mrp'], equals(originalData['mrp']));
        expect(map['category'], equals(originalData['category']));
      });
    });

    group('discountPercentage', () {
      test('should calculate correct discount percentage', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 75.0,
          mrp: 100.0,
          stock: 50,
        );

        expect(medicine.discountPercentage, equals(25.0));
      });

      test('should return 0 when price >= mrp', () {
        final medicine1 = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 100.0,
          mrp: 100.0,
          stock: 50,
        );

        final medicine2 = MedicineModel(
          id: 'med-002',
          name: 'Medicine',
          category: 'General',
          price: 120.0,
          mrp: 100.0,
          stock: 50,
        );

        expect(medicine1.discountPercentage, equals(0.0));
        expect(medicine2.discountPercentage, equals(0.0));
      });

      test('should return 0 when mrp is 0', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 50.0,
          mrp: 0.0,
          stock: 50,
        );

        expect(medicine.discountPercentage, equals(0.0));
      });
    });

    group('isInStock', () {
      test('should return true when stock > 0', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 50.0,
          mrp: 100.0,
          stock: 1,
        );

        expect(medicine.isInStock, isTrue);
      });

      test('should return false when stock is 0', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 50.0,
          mrp: 100.0,
          stock: 0,
        );

        expect(medicine.isInStock, isFalse);
      });

      test('should return false when stock is negative', () {
        final medicine = MedicineModel(
          id: 'med-001',
          name: 'Medicine',
          category: 'General',
          price: 50.0,
          mrp: 100.0,
          stock: -5,
        );

        expect(medicine.isInStock, isFalse);
      });
    });
  });
}
