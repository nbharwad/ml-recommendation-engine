# Sushrut Aushadhi - Setup Guide

## Prerequisites

1. **Flutter SDK** - Install from https://flutter.dev
2. **Android Studio** - For Android development
3. **Firebase Account** - https://console.firebase.google.com

---

## Step 1: Firebase Setup

### 1.1 Create Firebase Project
1. Go to https://console.firebase.google.com
2. Click "Add project"
3. Name: `sushrut-aushadhi`
4. Disable Google Analytics (for MVP)
5. Click "Create project"

### 1.2 Enable Services
- **Authentication**: Sign-in method → Phone → Enable
- **Firestore Database**: Create database → Start in test mode → Region: asia-south1
- **Storage**: Get started → Start in test mode → Region: asia-south1

### 1.3 Configure Android App
1. In Firebase Console → Project Overview → Add Android app
2. Package name: `com.example.sushrut_aushadhi`
3. App nickname: `Sushrut Aushadhi`
4. Click "Register app"
5. Download `google-services.json`

### 1.4 Add SHA-1 Fingerprint (Required for Phone Auth)
```bash
cd android
./gradlew signingReport
```
Copy SHA-1 → Firebase Console → Project Settings → Add fingerprint

---

## Step 2: Project Configuration

### 2.1 Replace google-services.json
Move your downloaded `google-services.json` to:
```
android/app/google-services.json
```

### 2.2 Configure Firebase CLI
```bash
# Install FlutterFire CLI (if not installed)
dart pub global activate flutterfire_cli

# Navigate to project
cd sushrut_aushadhi

# Configure Firebase
flutterfire configure --project=sushrut-aushadhi
```

### 2.3 Update local.properties
Edit `android/local.properties` and update the Flutter SDK path:
```
flutter.sdk=/path/to/your/flutter/sdk
```

---

## Step 3: Install Dependencies

```bash
cd sushrut_aushadhi
flutter pub get
```

---

## Step 4: Run the App

### Debug Build
```bash
flutter run
```

### Release Build
```bash
flutter build apk --release
```

---

## Step 5: Make Yourself Admin

1. Open the app and login with your phone number
2. Go to Firebase Console → Firestore → users
3. Find your user document (your UID)
4. Click "Edit" and add field:
   - Field: `isAdmin`
   - Type: `boolean`
   - Value: `true`
5. Restart the app

---

## Step 6: Add Seed Data (Optional)

Add sample medicines to Firestore `medicines` collection:

```json
{
  "name": "Paracetamol 500mg",
  "genericName": "Paracetamol",
  "manufacturer": "Cipla",
  "category": "fever",
  "price": 15.50,
  "mrp": 18.00,
  "stock": 100,
  "unit": "strip",
  "requiresPrescription": false,
  "isActive": true
}
```

---

## Troubleshooting

### Issue: Phone OTP not working
- ✅ Verify SHA-1 fingerprint is added to Firebase
- ✅ Check package name matches in google-services.json

### Issue: App crashes on launch
- ✅ Check google-services.json is in correct location
- ✅ Verify Firebase services are enabled

### Issue: Build errors
- ✅ Run `flutter clean` first
- ✅ Verify Flutter SDK path in local.properties
- ✅ Check Android SDK is properly configured

---

## Project Structure

```
sushrut_aushadhi/
├── android/                 # Android configuration
│   └── app/
│       └── google-services.json  # YOUR FIREBASE CONFIG
├── lib/
│   ├── main.dart           # Entry point
│   ├── core/               # Constants, routes, utils
│   ├── models/             # Data models
│   ├── services/           # Firebase services
│   ├── providers/          # Riverpod state
│   └── features/           # Screens
├── pubspec.yaml            # Dependencies
└── README.md               # Project info
```

---

## Build Commands

| Command | Description |
|---------|-------------|
| `flutter clean` | Clean build cache |
| `flutter pub get` | Install dependencies |
| `flutter analyze` | Check for errors |
| `flutter build apk --debug` | Debug APK |
| `flutter build apk --release` | Release APK |
| `flutter build appbundle` | App bundle for Play Store |
