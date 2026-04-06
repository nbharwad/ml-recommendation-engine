import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/widgets/main_shell.dart';
import '../../features/admin/admin_lab_orders_screen.dart';
import '../../features/admin/admin_lab_packages_screen.dart';
import '../../features/admin/admin_lab_tests_screen.dart';
import '../../features/admin/admin_order_detail_screen.dart';
import '../../features/admin/admin_shell.dart';
import '../../features/lab/lab_order_detail_screen.dart';
import '../../features/lab/lab_order_request_screen.dart';
import '../../features/lab/lab_orders_screen.dart';
import '../../features/lab/lab_package_detail_screen.dart';
import '../../features/auth/login_screen.dart';
import '../../features/auth/otp_screen.dart';
import '../../features/cart/cart_screen.dart';
import '../../features/home/home_screen.dart';
import '../../features/home/search_screen.dart';
import '../../features/medicine/medicine_detail_screen.dart';
import '../../features/orders/order_detail_screen.dart';
import '../../features/orders/order_confirmation_screen.dart';
import '../../features/orders/orders_screen.dart';
import '../../features/prescription/my_prescriptions_screen.dart';
import '../../features/prescription/prescription_upload_screen.dart';
import '../../models/prescription_model.dart';
import '../../features/profile/profile_screen.dart';
import '../../features/splash/splash_screen.dart';
import '../../providers/auth_provider.dart';

class GoRouterRefreshStream extends ChangeNotifier {
  GoRouterRefreshStream(Stream<dynamic> stream) {
    notifyListeners();
    _subscription = stream.asBroadcastStream().listen((_) => notifyListeners());
  }

  late final StreamSubscription<dynamic> _subscription;

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }
}

final isVerifyingProvider = StateProvider<bool>((ref) => false);

final goRouterProvider = Provider<GoRouter>((ref) {
  final isVerifying = ref.watch(isVerifyingProvider);
  final authListener =
      GoRouterRefreshStream(FirebaseAuth.instance.authStateChanges());

  return GoRouter(
    initialLocation: '/splash',
    refreshListenable: authListener,
    redirect: (context, state) {
      if (isVerifying) return null;

      final user = FirebaseAuth.instance.currentUser;
      final location = state.matchedLocation;

      final isAuthRoute = location == '/login' || location == '/otp';

      final isProtected = location.startsWith('/order/') ||
          location.startsWith('/prescription') ||
          location.startsWith('/admin') ||
          location.startsWith('/lab/orders') ||
          location.startsWith('/lab-order/');

      if (user != null && isAuthRoute) return '/home';

      if (user == null && isProtected) return '/login';

      if (location.startsWith('/admin')) {
        final adminAuthState = ref.read(adminAuthStateProvider).valueOrNull;

        // Allow while loading - let admin shell show loading state
        if (adminAuthState == null ||
            adminAuthState.status == AdminAuthStatus.loading) {
          return null;
        }

        // Redirect only for notAdmin
        if (adminAuthState.status == AdminAuthStatus.notAdmin) {
          return '/home';
        }

        // For admin and error: let admin shell handle it
      }

      return null;
    },
    routes: [
      GoRoute(
        path: '/splash',
        builder: (context, state) => const SplashScreen(),
      ),
      GoRoute(
        path: '/login',
        builder: (context, state) => const LoginScreen(),
      ),
      GoRoute(
        path: '/otp',
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>?;
          final verificationId = extra?['verificationId'] as String?;
          final phoneNumber = extra?['phoneNumber'] as String?;
          if (verificationId == null ||
              verificationId.isEmpty ||
              phoneNumber == null ||
              phoneNumber.isEmpty) {
            return const LoginScreen();
          }
          return OtpScreen(
            verificationId: verificationId,
            phoneNumber: phoneNumber,
          );
        },
      ),
      StatefulShellRoute.indexedStack(
        builder: (context, state, navigationShell) =>
            MainShell(navigationShell: navigationShell),
        branches: [
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/home',
                builder: (context, state) => const HomeScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/search',
                builder: (context, state) => const SearchScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/cart',
                builder: (context, state) {
                  final fromProfile =
                      state.uri.queryParameters['fromProfile'] == 'true';
                  return CartScreen(fromProfile: fromProfile);
                },
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/orders',
                builder: (context, state) => const OrdersScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: '/profile',
                builder: (context, state) {
                  final redirectTo = state.uri.queryParameters['redirectTo'];
                  return ProfileScreen(redirectTo: redirectTo);
                },
              ),
            ],
          ),
        ],
      ),
      GoRoute(
        path: '/medicine',
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>?;
          final hasDirectMedicinePayload = extra != null &&
              !extra.containsKey('medicine') &&
              !extra.containsKey('localMedicine') &&
              !extra.containsKey('medicineId');
          return MedicineDetailScreen(
            medicineId:
                extra?['medicineId'] as String? ?? extra?['id'] as String?,
            medicine: hasDirectMedicinePayload
                ? extra
                : extra?['medicine'] as Map<String, dynamic>?,
            localMedicine: extra?['localMedicine'] as Map<String, dynamic>?,
          );
        },
      ),
      GoRoute(
        path: '/medicine/:id',
        builder: (context, state) {
          final medicineId = state.pathParameters['id'];
          final extra = state.extra as Map<String, dynamic>?;

          if (medicineId != null && medicineId.isNotEmpty) {
            return MedicineDetailScreen(
              medicineId: medicineId,
              medicine: extra?['medicine'] as Map<String, dynamic>?,
              localMedicine: extra?['localMedicine'] as Map<String, dynamic>?,
            );
          }
          return const HomeScreen();
        },
      ),
      GoRoute(
        path: '/prescription',
        builder: (context, state) {
          final typeParam = state.uri.queryParameters['type'];
          final prescriptionType = PrescriptionType.fromString(typeParam);
          return PrescriptionUploadScreen(prescriptionType: prescriptionType);
        },
      ),
      GoRoute(
        path: '/my-prescriptions',
        builder: (context, state) {
          final typeParam = state.uri.queryParameters['type'];
          return MyPrescriptionsScreen(typeFilter: typeParam);
        },
      ),
      GoRoute(
        path: '/order/:id',
        builder: (context, state) {
          final orderId = state.pathParameters['id'];
          if (orderId == null || orderId.isEmpty) {
            return const OrdersScreen();
          }
          return OrderDetailScreen(orderId: orderId);
        },
      ),
      GoRoute(
        path: '/order-confirmation',
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>?;
          if (extra == null) {
            return const HomeScreen();
          }
          return OrderConfirmationScreen(
            orderId: extra['orderId'] as String? ?? '',
            totalAmount: (extra['totalAmount'] as num?)?.toDouble() ?? 0.0,
            items:
                (extra['items'] as List?)?.cast<Map<String, dynamic>>() ?? [],
            deliveryAddress: extra['deliveryAddress'] as String? ?? '',
          );
        },
      ),
      GoRoute(
        path: '/admin',
        builder: (context, state) => const AdminShellScreen(),
      ),
      GoRoute(
        path: '/admin/order/:id',
        builder: (context, state) {
          final orderId = state.pathParameters['id'];
          if (orderId == null || orderId.isEmpty) {
            return const AdminShellScreen();
          }
          return AdminOrderDetailScreen(orderId: orderId);
        },
      ),
      GoRoute(
        path: '/admin/prescriptions',
        redirect: (context, state) => '/admin',
      ),
      // Lab Test routes — fix existing bug + new screens
      GoRoute(
        path: '/lab-order/:id',
        builder: (context, state) {
          final orderId = state.pathParameters['id'];
          if (orderId == null || orderId.isEmpty) return const HomeScreen();
          return LabOrderDetailScreen(orderId: orderId);
        },
      ),
      GoRoute(
        path: '/lab/package/:id',
        builder: (context, state) {
          final packageId = state.pathParameters['id'];
          if (packageId == null || packageId.isEmpty) return const HomeScreen();
          return LabPackageDetailScreen(packageId: packageId);
        },
      ),
      GoRoute(
        path: '/lab/book',
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>?;
          return LabOrderRequestScreen(
            packageId: extra?['packageId'] as String?,
            preselectedTestIds: (extra?['testIds'] as List?)
                    ?.map((e) => e.toString())
                    .toList() ??
                [],
            packageName: extra?['packageName'] as String?,
            packagePrice: (extra?['packagePrice'] as num?)?.toDouble(),
          );
        },
      ),
      GoRoute(
        path: '/lab/orders',
        builder: (context, state) => const LabOrdersScreen(),
      ),
      GoRoute(
        path: '/admin/lab-orders',
        builder: (context, state) => const AdminLabOrdersScreen(),
      ),
      GoRoute(
        path: '/admin/lab-order/:id',
        builder: (context, state) {
          final orderId = state.pathParameters['id'];
          if (orderId == null || orderId.isEmpty)
            return const AdminLabOrdersScreen();
          return LabOrderDetailScreen(orderId: orderId);
        },
      ),
      GoRoute(
        path: '/admin/lab-packages',
        builder: (context, state) => const AdminLabPackagesScreen(),
      ),
      GoRoute(
        path: '/admin/lab-tests',
        builder: (context, state) => const AdminLabTestsScreen(),
      ),
    ],
  );
});
