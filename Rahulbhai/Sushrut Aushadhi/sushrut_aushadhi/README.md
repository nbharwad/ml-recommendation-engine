# Sushrut Aushadhi

A Flutter + Firebase medicine ordering app for pharmacies.

## Features

- Phone OTP authentication
- Browse medicines by category
- Search medicines
- Add to cart functionality
- Prescription upload for regulated medicines
- Order tracking
- Admin panel for order management
- Push notifications

## Setup Instructions

### 1. Prerequisites
- Flutter SDK 3.0+
- Node.js (for Firebase CLI)
- Firebase project

### 2. Firebase Setup

1. Create a Firebase project at https://console.firebase.google.com
2. Enable Authentication → Phone sign-in
3. Create Firestore database (start in test mode)
4. Enable Firebase Storage

### 3. Configure the App

```bash
# Navigate to project directory
cd sushrut_aushadhi

# Install Flutter dependencies
flutter pub get

# Configure Firebase (this will generate firebase_options.dart)
flutterfire configure --project=sushrut-aushadhi

# Run the app
flutter run
```

### 4. Add Seed Data

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

### 5. Make Admin User

After first login, manually set `isAdmin: true` in Firestore:
- Collection: `users`
- Document: `{your_uid}`
- Field: `isAdmin` = true

## Project Structure

```
lib/
├── core/           # Constants, routes, utils, widgets
├── models/         # Data models
├── services/       # Firebase services
├── providers/      # Riverpod providers
└── features/      # Screens by feature
```

## Build APK

```bash
# Debug build
flutter build apk --debug

# Release build
flutter build apk --release

# App bundle (for Play Store)
flutter build appbundle --release
```

## Firestore Security Rules

Deploy security rules from the blueprint document to protect your data.

## License

MIT License
