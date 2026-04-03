import 'package:cloud_firestore/cloud_firestore.dart';

Future<void> seedLabTests() async {
  final db = FirebaseFirestore.instance;
  final tests = [
    {'name': 'Complete Blood Count (CBC)', 'category': 'Haematology', 'price': 350.0, 'tat_hours': 24, 'sample_type': 'Blood', 'active': true},
    {'name': 'Blood Sugar (Fasting)', 'category': 'Biochemistry', 'price': 200.0, 'tat_hours': 12, 'sample_type': 'Blood', 'active': true},
    {'name': 'Lipid Profile', 'category': 'Biochemistry', 'price': 600.0, 'tat_hours': 24, 'sample_type': 'Blood', 'active': true},
    {'name': 'Thyroid Profile (TSH)', 'category': 'Biochemistry', 'price': 800.0, 'tat_hours': 24, 'sample_type': 'Blood', 'active': true},
    {'name': 'Liver Function Test', 'category': 'Biochemistry', 'price': 700.0, 'tat_hours': 24, 'sample_type': 'Blood', 'active': true},
    {'name': 'Kidney Function Test', 'category': 'Biochemistry', 'price': 650.0, 'tat_hours': 24, 'sample_type': 'Blood', 'active': true},
    {'name': 'HbA1c', 'category': 'Biochemistry', 'price': 400.0, 'tat_hours': 24, 'sample_type': 'Blood', 'active': true},
    {'name': 'Urine Routine', 'category': 'Pathology', 'price': 200.0, 'tat_hours': 12, 'sample_type': 'Urine', 'active': true},
    {'name': 'Blood Pressure Test', 'category': 'Clinical', 'price': 150.0, 'tat_hours': 1, 'sample_type': 'None', 'active': true},
  ];

  final batch = db.batch();
  for (final test in tests) {
    final ref = db.collection('lab_tests').doc();
    batch.set(ref, {...test, 'createdAt': FieldValue.serverTimestamp()});
  }
  await batch.commit();
  print('Lab tests seeded successfully');
}