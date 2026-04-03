import 'package:flutter_test/flutter_test.dart';

import 'package:sushrut_aushadhi/services/rate_limit_service.dart';

void main() {
  group('RateLimitService Tests', () {
    test('Rate limits have correct constants defined', () {
      expect(RateLimitService.maxOrdersPerDay, equals(10));
      expect(RateLimitService.maxPrescriptionsPerDay, equals(5));
      expect(RateLimitService.maxSearchPerMinute, equals(30));
    });
  });
}