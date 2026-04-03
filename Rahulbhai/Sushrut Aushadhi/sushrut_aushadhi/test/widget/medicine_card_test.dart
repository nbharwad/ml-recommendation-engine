import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sushrut_aushadhi/features/home/widgets/medicine_card.dart';
import 'package:sushrut_aushadhi/models/medicine_model.dart';

void main() {
  group('MedicineCard Widget Tests', () {
    late MedicineModel testMedicine;

    setUp(() {
      testMedicine = MedicineModel(
        id: '1',
        name: 'Dolo 650',
        genericName: 'Paracetamol',
        manufacturer: 'Micro Labs',
        category: 'Pain Relief',
        price: 35.0,
        mrp: 45.0,
        stock: 50,
        unit: 'strip',
        imageUrl: '',
        requiresPrescription: false,
        description: 'Pain relief tablet',
        isActive: true,
      );
    });

    testWidgets('displays medicine name', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: testMedicine,
                onTap: () {},
              ),
            ),
          ),
        ),
      );

      expect(find.text('Dolo 650'), findsOneWidget);
    });

    testWidgets('displays generic name', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: testMedicine,
                onTap: () {},
              ),
            ),
          ),
        ),
      );

      expect(find.text('Paracetamol'), findsOneWidget);
    });

    testWidgets('displays Rx badge when prescription required', (WidgetTester tester) async {
      final prescriptionMedicine = MedicineModel(
        id: '2',
        name: 'Amoxicillin',
        genericName: 'Amoxicillin',
        manufacturer: 'Cipla',
        category: 'Antibiotic',
        price: 150.0,
        mrp: 180.0,
        stock: 30,
        unit: 'strip',
        requiresPrescription: true,
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: prescriptionMedicine,
                onTap: () {},
              ),
            ),
          ),
        ),
      );

      expect(find.text('Rx'), findsOneWidget);
    });

    testWidgets('displays discount percentage when discount available', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: testMedicine,
                onTap: () {},
              ),
            ),
          ),
        ),
      );

      expect(find.text('22%'), findsOneWidget);
    });

    testWidgets('displays out of stock when not in stock', (WidgetTester tester) async {
      final outOfStockMedicine = MedicineModel(
        id: '3',
        name: 'Unavailable Med',
        genericName: '',
        manufacturer: '',
        category: 'Test',
        price: 100.0,
        mrp: 100.0,
        stock: 0,
        unit: 'strip',
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: outOfStockMedicine,
                onTap: () {},
              ),
            ),
          ),
        ),
      );

      expect(find.text('Out of Stock'), findsOneWidget);
    });

    testWidgets('calls onTap when tapped', (WidgetTester tester) async {
      bool tapped = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: testMedicine,
                onTap: () => tapped = true,
              ),
            ),
          ),
        ),
      );

      await tester.tap(find.byType(InkWell));
      expect(tapped, isTrue);
    });

    testWidgets('displays default medication icon when no image', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 150,
              height: 200,
              child: MedicineCard(
                medicine: testMedicine,
                onTap: () {},
              ),
            ),
          ),
        ),
      );

      expect(find.byIcon(Icons.medication), findsOneWidget);
    });
  });
}
