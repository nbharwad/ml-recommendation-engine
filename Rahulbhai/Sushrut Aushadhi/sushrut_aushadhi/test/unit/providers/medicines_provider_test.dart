import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:sushrut_aushadhi/providers/medicines_provider.dart';

void main() {
  group('Medicines Provider Tests', () {
    late ProviderContainer container;

    setUp(() {
      container = ProviderContainer();
    });

    tearDown(() {
      container.dispose();
    });

    test('selectedCategoryProvider has default value "all"', () {
      final category = container.read(selectedCategoryProvider);
      expect(category, equals('all'));
    });

    test('selectedCategoryProvider can be updated', () {
      container.read(selectedCategoryProvider.notifier).state = 'Pain Relief';
      final category = container.read(selectedCategoryProvider);
      expect(category, equals('Pain Relief'));
    });

    test('searchQueryProvider has default empty string', () {
      final query = container.read(searchQueryProvider);
      expect(query, equals(''));
    });

    test('searchQueryProvider can be updated', () {
      container.read(searchQueryProvider.notifier).state = 'Dolo';
      final query = container.read(searchQueryProvider);
      expect(query, equals('Dolo'));
    });

    test('searchQueryProvider can be cleared', () {
      container.read(searchQueryProvider.notifier).state = 'Dolo';
      container.read(searchQueryProvider.notifier).state = '';
      final query = container.read(searchQueryProvider);
      expect(query, equals(''));
    });
  });
}
