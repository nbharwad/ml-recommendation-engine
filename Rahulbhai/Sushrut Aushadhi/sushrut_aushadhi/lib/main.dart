import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:firebase_app_check/firebase_app_check.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/constants/app_strings.dart';
import 'core/constants/app_theme.dart';
import 'core/routes/app_router.dart' show goRouterProvider;
import 'core/utils/app_logger.dart';
import 'core/widgets/connectivity_wrapper.dart';
import 'firebase_options.dart';
import 'providers/notification_handler_provider.dart';
import 'services/connectivity_service.dart';
import 'services/medicine_cache_service.dart';
import 'services/remote_config_service.dart';

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  AppLogger.init();

  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

  // Initialize Crashlytics
  await FirebaseCrashlytics.instance.setCrashlyticsCollectionEnabled(true);
  
  // Pass all uncaught errors from the Flutter framework to Crashlytics
  FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterError;

  // Initialize App Check with Play Integrity
  await FirebaseAppCheck.instance.activate(
    androidProvider: kDebugMode ? AndroidProvider.debug : AndroidProvider.playIntegrity,
    appleProvider: AppleProvider.appAttest,
  );

  FirebaseFirestore.instance.settings = const Settings(
    persistenceEnabled: true,
    cacheSizeBytes: 100 * 1024 * 1024, // 100MB max
  );

  MedicineCacheService.loadCache();
  await ConnectivityService.initialize();
  await RemoteConfigService.initialize();

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

class SushrutAushadhiApp extends ConsumerStatefulWidget {
  const SushrutAushadhiApp({super.key});

  @override
  ConsumerState<SushrutAushadhiApp> createState() => _SushrutAushadhiAppState();
}

class _SushrutAushadhiAppState extends ConsumerState<SushrutAushadhiApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeNotifications();
    });
  }

  Future<void> _initializeNotifications() async {
    final router = ref.read(goRouterProvider);
    await ref.read(notificationHandlerProvider.notifier).initialize(router);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      RemoteConfigService.refresh();
    }
  }

  @override
  Widget build(BuildContext context) {
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
