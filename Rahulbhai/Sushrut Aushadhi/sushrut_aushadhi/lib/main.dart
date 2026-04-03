import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:firebase_app_check/firebase_app_check.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/constants/app_strings.dart';
import 'core/constants/app_theme.dart';
import 'core/routes/app_router.dart' show goRouterProvider;
import 'core/widgets/connectivity_wrapper.dart';
import 'firebase_options.dart';
import 'services/connectivity_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  // Initialize Crashlytics
  await FirebaseCrashlytics.instance.setCrashlyticsCollectionEnabled(true);
  
  // Pass all uncaught errors from the Flutter framework to Crashlytics
  FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterError;

  // Initialize App Check with Play Integrity
  await FirebaseAppCheck.instance.activate(
    androidProvider: AndroidProvider.playIntegrity,
    appleProvider: AppleProvider.appAttest,
  );

  FirebaseFirestore.instance.settings = const Settings(
    persistenceEnabled: true,
    cacheSizeBytes: Settings.CACHE_SIZE_UNLIMITED,
  );

  await ConnectivityService.initialize();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Run the app with error zone handling
  runZonedGuarded<Future<void>>(() async {
    runApp(const ProviderScope(child: SushrutAushadhiApp()));
  }, (error, stack) {
    FirebaseCrashlytics.instance.recordError(error, stack);
  });
}

class SushrutAushadhiApp extends ConsumerWidget {
  const SushrutAushadhiApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(goRouterProvider);
    return MaterialApp.router(
      title: AppStrings.appName,
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      routerConfig: router,
      builder: (context, child) => ConnectivityWrapper(
        child: child ?? const SizedBox(),
      ),
    );
  }
}
