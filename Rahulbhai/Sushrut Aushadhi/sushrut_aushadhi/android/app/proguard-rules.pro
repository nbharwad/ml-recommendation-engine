# Flutter
-keep class io.flutter.** { *; }
-keep class io.flutter.plugins.** { *; }

# Firebase
-keep class com.google.firebase.** { *; }
-keep class com.google.android.gms.** { *; }

# Kotlin
-keep class kotlin.** { *; }
-keep class kotlinx.** { *; }

# Image picker
-keep class io.flutter.plugins.imagepicker.** { *; }

# Permission handler
-keep class com.baseflow.permissionhandler.** { *; }

# Keep model classes
-keep class com.example.sushrut_aushadhi.** { *; }

# Prevent stripping of crash reporting
-keepattributes SourceFile,LineNumberTable
-keep public class * extends java.lang.Exception

# Google Play Core (required by Flutter)
-keep class com.google.android.play.core.** { *; }

# Firebase Crashlytics
-keep class com.google.firebase.crashlytics.** { *; }
-dontwarn com.google.firebase.crashlytics.**

# Go Router
-keep class go.** { *; }

# Riverpod
-keep class riverpod.** { *; }
-keep class flutter_riverpod.** { *; }
-keep class hooks_riverpod.** { *; }