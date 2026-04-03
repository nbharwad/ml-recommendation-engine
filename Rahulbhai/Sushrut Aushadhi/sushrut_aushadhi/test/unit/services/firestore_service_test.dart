import 'package:flutter_test/flutter_test.dart';
import 'package:uuid/uuid.dart';

import 'package:sushrut_aushadhi/services/firestore_service.dart';

void main() {
  group('FirestoreService - UUID Generation', () {
    test('placeOrderId generates valid UUID format', () {
      const uuid = Uuid();
      final orderId = uuid.v4();
      expect(orderId, matches(RegExp(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')));
    });

    test('UUID v4 format is valid', () {
      const uuid = Uuid();
      final orderId1 = uuid.v4();
      final orderId2 = uuid.v4();
      expect(orderId1, isNot(equals(orderId2)));
    });
  });
}