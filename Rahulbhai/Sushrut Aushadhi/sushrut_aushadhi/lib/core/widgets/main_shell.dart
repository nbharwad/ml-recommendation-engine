import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../constants/app_colors.dart';
import '../../providers/cart_provider.dart';
import '../../providers/notification_provider.dart';

class MainShell extends ConsumerWidget {
  final StatefulNavigationShell navigationShell;

  const MainShell({
    super.key,
    required this.navigationShell,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final cartCount = ref.watch(cartItemCountProvider);
    final unreadCount = ref.watch(unreadCountProvider);
    final bottomPadding = MediaQuery.of(context).padding.bottom;
    final screenHeight = MediaQuery.of(context).size.height;
    
    // Responsive height: base 64 + bottom padding, scales with screen
    final navHeight = 64 + bottomPadding + (screenHeight > 800 ? 4 : 0);

    return Scaffold(
      body: navigationShell,
      bottomNavigationBar: NavigationBarTheme(
        data: NavigationBarThemeData(
          // Pill-shaped (stadium) indicator — the Material 3 spec default
          indicatorShape: const StadiumBorder(),
          indicatorColor: AppColors.primaryLight,
          // Always show labels so the pill + label combo feels complete
          labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
          // Icon colours
          iconTheme: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) {
              return const IconThemeData(color: AppColors.primary, size: 24);
            }
            return const IconThemeData(color: AppColors.textSecondary, size: 24);
          }),
          // Label text style
          labelTextStyle: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) {
              return const TextStyle(
                color: AppColors.primary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.2,
              );
            }
            return const TextStyle(
              color: AppColors.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w400,
            );
          }),
          backgroundColor: Colors.white,
          elevation: 8,
          shadowColor: Colors.black12,
          surfaceTintColor: Colors.transparent,
          height: navHeight,
        ),
        child: NavigationBar(
          selectedIndex: navigationShell.currentIndex,
          onDestinationSelected: (index) {
            navigationShell.goBranch(
              index,
              initialLocation: index == navigationShell.currentIndex,
            );
          },
          destinations: [
            const NavigationDestination(
              icon: Icon(Icons.home_outlined),
              selectedIcon: Icon(Icons.home),
              label: 'Home',
            ),
            const NavigationDestination(
              icon: Icon(Icons.search_outlined),
              selectedIcon: Icon(Icons.search),
              label: 'Search',
            ),
            NavigationDestination(
              icon: cartCount > 0
                  ? Badge.count(count: cartCount, child: const Icon(Icons.shopping_cart_outlined))
                  : const Icon(Icons.shopping_cart_outlined),
              selectedIcon: cartCount > 0
                  ? Badge.count(count: cartCount, child: const Icon(Icons.shopping_cart))
                  : const Icon(Icons.shopping_cart),
              label: 'Cart',
            ),
            const NavigationDestination(
              icon: Icon(Icons.receipt_long_outlined),
              selectedIcon: Icon(Icons.receipt_long),
              label: 'Orders',
            ),
            NavigationDestination(
              icon: unreadCount > 0
                  ? Badge.count(count: unreadCount, child: const Icon(Icons.person_outline))
                  : const Icon(Icons.person_outline),
              selectedIcon: unreadCount > 0
                  ? Badge.count(count: unreadCount, child: const Icon(Icons.person))
                  : const Icon(Icons.person),
              label: 'Profile',
            ),
          ],
        ),
      ),
    );
  }
}
