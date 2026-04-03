import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/connectivity_provider.dart';
import 'no_internet_banner.dart';

class ConnectivityWrapper extends ConsumerStatefulWidget {
  final Widget child;

  const ConnectivityWrapper({super.key, required this.child});

  @override
  ConsumerState<ConnectivityWrapper> createState() => _ConnectivityWrapperState();
}

class _ConnectivityWrapperState extends ConsumerState<ConnectivityWrapper> {
  bool _showBackOnline = false;
  bool _wasOffline = false;
  Timer? _backOnlineTimer;

  @override
  void initState() {
    super.initState();
    ref.listenManual(connectivityProvider, (prev, next) {
      next.whenData((isOnline) {
        if (isOnline && _wasOffline) {
          setState(() => _showBackOnline = true);
          _backOnlineTimer?.cancel();
          _backOnlineTimer = Timer(const Duration(seconds: 3), () {
            if (mounted) {
              setState(() => _showBackOnline = false);
            }
          });
        }
        _wasOffline = !isOnline;
      });
    });
  }

  @override
  void dispose() {
    _backOnlineTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final connectivityAsync = ref.watch(connectivityProvider);
    final isOnline = connectivityAsync.asData?.value ?? true;

    return Column(
      children: [
        if (!isOnline) const NoInternetBanner(),
        if (_showBackOnline && isOnline) const BackOnlineBanner(),
        Expanded(child: widget.child),
      ],
    );
  }
}
