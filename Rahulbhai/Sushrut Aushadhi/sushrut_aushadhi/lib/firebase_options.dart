import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

/// Default Firebase configuration options for your app.
///
/// To configure Firebase for your project:
/// 1. Go to https://console.firebase.google.com
/// 2. Create a new project named "sushrut-aushadhi"
/// 3. Run: flutterfire configure --project=sushrut-aushadhi
/// 4. This file will be auto-generated with your Firebase config

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      throw UnsupportedError(
        'Firebase Web support has not been configured yet.\n'
        'Please run `flutterfire configure` to generate firebase_options.dart',
      );
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        throw UnsupportedError(
          'Firebase macOS support has not been configured yet.\n'
          'Please run `flutterfire configure` to generate firebase_options.dart',
        );
      case TargetPlatform.windows:
        throw UnsupportedError(
          'Firebase Windows support has not been configured yet.\n'
          'Please run `flutterfire configure` to generate firebase_options.dart',
        );
      case TargetPlatform.linux:
        throw UnsupportedError(
          'Firebase Linux support has not been configured yet.\n'
          'Please run `flutterfire configure` to generate firebase_options.dart',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for platform - ${defaultTargetPlatform.name}.\n'
          'Please run `flutterfire configure` to generate firebase_options.dart',
        );
    }
  }

  // TODO: Replace these values with your Firebase project configuration

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyBh9arVnt7KO40CM6tWHkbOrQf0oq08PL8',
    appId: '1:803373581849:android:cf95a0a2684116049e4c88',
    messagingSenderId: '803373581849',
    projectId: 'sushrut-aushadhi',
    storageBucket: 'sushrut-aushadhi.firebasestorage.app',
  );

  // After running: flutterfire configure --project=sushrut-aushadhi

  // TODO: Replace these values with your Firebase project configuration
  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'YOUR_API_KEY',
    appId: '1:000000000000:ios:0000000000000000000000',
    messagingSenderId: '000000000000',
    projectId: 'sushrut-aushadhi',
    storageBucket: 'sushrut-aushadhi.appspot.com',
    iosBundleId: 'com.example.sushrutAushadhi',
  );
}