import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:sushrut_aushadhi/providers/auth_provider.dart';
import 'package:sushrut_aushadhi/models/user_model.dart';

void main() {
  group('isAdminProvider Tests', () {
    late ProviderContainer container;

    setUp(() {
      container = ProviderContainer();
    });

    tearDown(() {
      container.dispose();
    });

    test('isAdminProvider returns false when user is null', () {
      final isAdmin = container.read(isAdminProvider);
      expect(isAdmin, equals(false));
    });

    test('isAdminProvider returns false when user is not admin', () {
      final isAdmin = container.read(isAdminProvider);
      expect(isAdmin, equals(false));
    });

    test('isAdminProvider returns false for non-admin user data', () {
      final isAdmin = container.read(isAdminProvider);
      expect(isAdmin, equals(false));
    });
  });
}