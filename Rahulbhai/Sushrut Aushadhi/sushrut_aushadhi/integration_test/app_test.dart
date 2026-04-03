import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:sushrut_aushadhi/main.dart' as app;
import 'package:sushrut_aushadhi/core/widgets/custom_button.dart';
import 'package:sushrut_aushadhi/core/widgets/custom_text_field.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('App Launch Test', () {
    testWidgets('App should launch without crashing', (WidgetTester tester) async {
      app.main();
      await tester.pump();
      await tester.pump(const Duration(seconds: 3));
      
      expect(find.byType(app.SushrutAushadhiApp), findsOneWidget);
    });

    testWidgets('App should show home screen after launch', (WidgetTester tester) async {
      app.main();
      await tester.pump();
      await tester.pump(const Duration(seconds: 3));
      
      expect(find.byType(app.SushrutAushadhiApp), findsOneWidget);
    });
  });

  group('Home Screen Test', () {
    testWidgets('App should navigate to home screen after splash', (WidgetTester tester) async {
      app.main();
      await tester.pump();
      await tester.pump(const Duration(seconds: 3));
      
      expect(find.text('Sushrut Aushadhi'), findsWidgets);
    });
  });

  group('Custom Button Widget Test', () {
    testWidgets('CustomButton renders with text', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: CustomButton(
              text: 'Test Button',
              onPressed: null,
            ),
          ),
        ),
      );

      expect(find.text('Test Button'), findsOneWidget);
    });

    testWidgets('CustomButton responds to tap', (WidgetTester tester) async {
      bool tapped = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: CustomButton(
              text: 'Tap Me',
              onPressed: () => tapped = true,
            ),
          ),
        ),
      );

      await tester.tap(find.byType(ElevatedButton));
      await tester.pump();

      expect(tapped, isTrue);
    });
  });

  group('Custom TextField Widget Test', () {
    testWidgets('CustomTextField renders with hint text', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: CustomTextField(
              hintText: 'Enter something',
            ),
          ),
        ),
      );

      expect(find.text('Enter something'), findsOneWidget);
    });

    testWidgets('CustomTextField accepts text input', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: CustomTextField(),
          ),
        ),
      );

      await tester.enterText(find.byType(TextFormField), 'Test input');
      expect(find.text('Test input'), findsOneWidget);
    });
  });
}
