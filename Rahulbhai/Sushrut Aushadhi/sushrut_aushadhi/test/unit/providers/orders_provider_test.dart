import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:sushrut_aushadhi/providers/orders_provider.dart';

void main() {
  group('Orders Provider Tests', () {
    late ProviderContainer container;

    setUp(() {
      container = ProviderContainer();
    });

    tearDown(() {
      container.dispose();
    });

    test('selectedStatusProvider has null default value', () {
      final status = container.read(selectedStatusProvider);
      expect(status, isNull);
    });

    test('selectedStatusProvider can be updated', () {
      container.read(selectedStatusProvider.notifier).state = 'pending';
      final status = container.read(selectedStatusProvider);
      expect(status, equals('pending'));
    });

    test('selectedStatusProvider can be cleared', () {
      container.read(selectedStatusProvider.notifier).state = 'pending';
      container.read(selectedStatusProvider.notifier).state = null;
      final status = container.read(selectedStatusProvider);
      expect(status, isNull);
    });

    test('selectedStatusProvider can be set to different statuses', () {
      container.read(selectedStatusProvider.notifier).state = 'pending';
      expect(container.read(selectedStatusProvider), equals('pending'));

      container.read(selectedStatusProvider.notifier).state = 'shipped';
      expect(container.read(selectedStatusProvider), equals('shipped'));

      container.read(selectedStatusProvider.notifier).state = 'delivered';
      expect(container.read(selectedStatusProvider), equals('delivered'));

      container.read(selectedStatusProvider.notifier).state = 'cancelled';
      expect(container.read(selectedStatusProvider), equals('cancelled'));
    });
  });
}
