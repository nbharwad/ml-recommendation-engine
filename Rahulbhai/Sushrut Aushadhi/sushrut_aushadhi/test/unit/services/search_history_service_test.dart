import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sushrut_aushadhi/services/search_history_service.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  group('SearchHistoryService', () {
    group('getHistory', () {
      test('should return empty list when no history exists', () async {
        final history = await SearchHistoryService.getHistory();
        expect(history, isEmpty);
      });

      test('should return existing search history', () async {
        SharedPreferences.setMockInitialValues({
          'search_history': '["paracetamol", "aspirin"]',
        });

        final history = await SearchHistoryService.getHistory();

        expect(history, equals(['paracetamol', 'aspirin']));
      });
    });

    group('addSearch', () {
      test('should add new search query to history', () async {
        await SearchHistoryService.addSearch('paracetamol');

        final history = await SearchHistoryService.getHistory();

        expect(history, contains('paracetamol'));
        expect(history.length, equals(1));
      });

      test('should add multiple searches in correct order', () async {
        await SearchHistoryService.addSearch('first');
        await SearchHistoryService.addSearch('second');
        await SearchHistoryService.addSearch('third');

        final history = await SearchHistoryService.getHistory();

        expect(history[0], equals('third'));
        expect(history[1], equals('second'));
        expect(history[2], equals('first'));
      });

      test('should move existing search to top', () async {
        await SearchHistoryService.addSearch('first');
        await SearchHistoryService.addSearch('second');
        await SearchHistoryService.addSearch('first');

        final history = await SearchHistoryService.getHistory();

        expect(history[0], equals('first'));
        expect(history.length, equals(2));
      });

      test('should not add empty or whitespace-only queries', () async {
        await SearchHistoryService.addSearch('');
        await SearchHistoryService.addSearch('   ');

        final history = await SearchHistoryService.getHistory();

        expect(history, isEmpty);
      });

      test('should trim whitespace from queries', () async {
        await SearchHistoryService.addSearch('  paracetamol  ');

        final history = await SearchHistoryService.getHistory();

        expect(history, contains('paracetamol'));
        expect(history, isNot(contains('  paracetamol  ')));
      });

      test('should limit history to 5 items', () async {
        await SearchHistoryService.addSearch('1');
        await SearchHistoryService.addSearch('2');
        await SearchHistoryService.addSearch('3');
        await SearchHistoryService.addSearch('4');
        await SearchHistoryService.addSearch('5');
        await SearchHistoryService.addSearch('6');

        final history = await SearchHistoryService.getHistory();

        expect(history.length, equals(5));
        expect(history[0], equals('6'));
        expect(history[4], equals('2'));
        expect(history, isNot(contains('1')));
      });
    });

    group('removeSearch', () {
      test('should remove specific search from history', () async {
        await SearchHistoryService.addSearch('paracetamol');
        await SearchHistoryService.addSearch('aspirin');
        await SearchHistoryService.addSearch('ibuprofen');

        await SearchHistoryService.removeSearch('aspirin');

        final history = await SearchHistoryService.getHistory();

        expect(history, equals(['ibuprofen', 'paracetamol']));
      });

      test('should handle removing non-existent search', () async {
        await SearchHistoryService.addSearch('paracetamol');

        await SearchHistoryService.removeSearch('nonexistent');

        final history = await SearchHistoryService.getHistory();

        expect(history, equals(['paracetamol']));
      });
    });

    group('clearHistory', () {
      test('should clear all search history', () async {
        await SearchHistoryService.addSearch('paracetamol');
        await SearchHistoryService.addSearch('aspirin');

        await SearchHistoryService.clearHistory();

        final history = await SearchHistoryService.getHistory();

        expect(history, isEmpty);
      });
    });
  });
}
