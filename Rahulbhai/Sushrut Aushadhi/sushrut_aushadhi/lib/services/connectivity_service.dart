import 'dart:async';
import 'package:connectivity_plus/connectivity_plus.dart';

class ConnectivityService {
  static final Connectivity _connectivity = Connectivity();
  static StreamController<bool> _controller = StreamController<bool>.broadcast();

  static Stream<bool> get onConnectivityChanged => _controller.stream;

  static bool _isOnline = true;
  static bool get isOnline => _isOnline;

  static Future<void> initialize() async {
    final result = await _connectivity.checkConnectivity();
    _isOnline = _isConnected(result);

    _connectivity.onConnectivityChanged.listen((ConnectivityResult result) {
      final wasOnline = _isOnline;
      _isOnline = _isConnected(result);

      if (wasOnline != _isOnline) {
        _controller.add(_isOnline);
      }
    });
  }

  static bool _isConnected(ConnectivityResult result) {
    return result == ConnectivityResult.mobile ||
        result == ConnectivityResult.wifi ||
        result == ConnectivityResult.ethernet;
  }

  static Future<bool> checkConnection() async {
    final result = await _connectivity.checkConnectivity();
    _isOnline = _isConnected(result);
    return _isOnline;
  }

  static void dispose() {
    _controller.close();
  }
}
