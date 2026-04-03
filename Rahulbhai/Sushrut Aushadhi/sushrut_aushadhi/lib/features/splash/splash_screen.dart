import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../providers/auth_provider.dart';
import '../../services/local_data_service.dart';
import '../../services/connectivity_service.dart';

class SplashScreen extends ConsumerStatefulWidget {
  const SplashScreen({super.key});

  @override
  ConsumerState<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends ConsumerState<SplashScreen>
    with SingleTickerProviderStateMixin {

  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.0, 0.6, curve: Curves.easeIn),
    ));

    _scaleAnimation = Tween<double>(
      begin: 0.7,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.0, 0.6, curve: Curves.elasticOut),
    ));

    _controller.forward();

    _initializeApp();
  }

  Future<void> _initializeApp() async {
    await Future.wait([
      LocalDataService.loadData(),
      ConnectivityService.initialize(),
      Future.delayed(const Duration(milliseconds: 2500)),
    ]);

    if (!mounted) return;

    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      // Force-refresh JWT so custom claims (role) are loaded before the router
      // evaluates the admin guard. Without this, returning admins get isAdmin=false.
      await user.getIdTokenResult(true);
      ref.invalidate(roleProvider);
    }

    // Wait until authReadyProvider is true (role future settled), max 5s.
    const maxWait = Duration(seconds: 5);
    final deadline = DateTime.now().add(maxWait);
    while (!ref.read(authReadyProvider)) {
      if (DateTime.now().isAfter(deadline)) break;
      await Future.delayed(const Duration(milliseconds: 50));
      if (!mounted) return;
    }

    if (!mounted) return;
    context.go(user != null ? '/home' : '/login');
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F6E56),
      body: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return FadeTransition(
            opacity: _fadeAnimation,
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ScaleTransition(
                    scale: _scaleAnimation,
                    child: Container(
                      width: 110,
                      height: 110,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.15),
                            blurRadius: 20,
                            offset: const Offset(0, 8),
                          ),
                        ],
                      ),
                      child: const Center(
                        child: Text('💊', style: TextStyle(fontSize: 54)),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    'Sushrut Aushadhi',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'सुश्रुत औषधि',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white.withOpacity(0.75),
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Your trusted medicine partner',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white.withOpacity(0.6),
                    ),
                  ),
                  const SizedBox(height: 60),
                  SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        Colors.white.withOpacity(0.7),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Loading medicines...',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withOpacity(0.5),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
