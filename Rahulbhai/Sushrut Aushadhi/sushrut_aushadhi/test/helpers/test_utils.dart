import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

Widget createTestableWidget(
  Widget child, {
  List<Override> overrides = const [],
}) {
  return ProviderScope(
    overrides: overrides,
    child: MaterialApp(
      home: Scaffold(body: child),
    ),
  );
}

Future<void> pumpUntilFound(
  WidgetTester tester,
  Finder finder, {
  Duration timeout = const Duration(seconds: 5),
}) async {
  final endTime = DateTime.now().add(timeout);
  while (DateTime.now().isBefore(endTime)) {
    await tester.pump(const Duration(milliseconds: 100));
    if (finder.evaluate().isNotEmpty) {
      return;
    }
  }
  throw TimeoutException('Widget not found within timeout', finder);
}

Future<void> waitForAnimation(WidgetTester tester, {int milliseconds = 300}) async {
  await tester.pump(Duration(milliseconds: milliseconds));
}

class TimeoutException implements Exception {
  final String message;
  final Finder? finder;
  TimeoutException(this.message, [this.finder]);
  
  @override
  String toString() => message;
}
