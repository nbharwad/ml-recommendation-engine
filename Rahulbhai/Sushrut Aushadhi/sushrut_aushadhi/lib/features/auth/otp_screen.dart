import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_strings.dart';
import '../../core/utils/responsive.dart';
import '../../core/widgets/custom_button.dart';
import '../../providers/auth_provider.dart';
import '../../services/auth_service.dart';

class OtpScreen extends ConsumerStatefulWidget {
  final String verificationId;
  final String phoneNumber;

  const OtpScreen({
    super.key,
    required this.verificationId,
    required this.phoneNumber,
  });

  @override
  ConsumerState<OtpScreen> createState() => _OtpScreenState();
}

class _OtpScreenState extends ConsumerState<OtpScreen> {
  final _authService = AuthService();
  final List<TextEditingController> _controllers =
      List.generate(6, (_) => TextEditingController());
  final List<FocusNode> _focusNodes = List.generate(6, (_) => FocusNode());

  late String _verificationId;
  bool _isLoading = false;
  String? _errorMessage;
  int _resendTimer = 60;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _verificationId = widget.verificationId;
    _startResendTimer();
  }

  @override
  void dispose() {
    for (final controller in _controllers) {
      controller.dispose();
    }
    for (final node in _focusNodes) {
      node.dispose();
    }
    _timer?.cancel();
    super.dispose();
  }

  void _startResendTimer() {
    _timer?.cancel();
    _resendTimer = 60;
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_resendTimer <= 1) {
        setState(() => _resendTimer = 0);
        timer.cancel();
        return;
      }
      setState(() => _resendTimer--);
    });
  }

  String get _otp => _controllers.map((c) => c.text).join();

  Future<void> _verifyOtp() async {
    if (_otp.length != 6) {
      setState(() => _errorMessage = 'Please enter 6-digit OTP');
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final user = await _authService.verifyOtp(
        verificationId: _verificationId,
        smsCode: _otp,
      );

      if (!mounted) {
        return;
      }
      if (user != null) {
        // Force-refresh the JWT token so custom claims (role) are available
        // before the router evaluates admin/customer redirect.
        await FirebaseAuth.instance.currentUser?.getIdTokenResult(true);
        ref.invalidate(roleProvider);
        if (!mounted) return;
        context.go('/home');
      } else {
        setState(() {
          _errorMessage = 'Failed to verify OTP';
          _isLoading = false;
        });
      }
    } catch (_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = 'Invalid OTP. Please try again.';
        _isLoading = false;
      });
    }
  }

  Future<void> _resendOtp() async {
    if (_resendTimer > 0) {
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    await _authService.sendOtp(
      phoneNumber: widget.phoneNumber,
      onCodeSent: (verificationId) {
        if (!mounted) {
          return;
        }
        setState(() {
          _verificationId = verificationId;
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('OTP sent successfully')),
        );
        _startResendTimer();
      },
      onVerificationCompleted: (_) {
        if (!mounted) {
          return;
        }
        setState(() => _isLoading = false);
        context.go('/cart');
      },
      onTimeout: (verificationId) {
        if (!mounted) {
          return;
        }
        setState(() {
          _verificationId = verificationId;
          _isLoading = false;
        });
      },
      onError: (error) {
        if (!mounted) {
          return;
        }
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $error')),
        );
      },
    );
  }

  void _onOtpChanged(int index, String value) {
    if (value.isNotEmpty && index < 5) {
      _focusNodes[index + 1].requestFocus();
    }
    if (value.isEmpty && index > 0) {
      _focusNodes[index - 1].requestFocus();
    }
  }

  Future<bool> _handleBackPressed() async {
    if (_isLoading) {
      return false;
    }
    context.go('/login');
    return false;
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: _handleBackPressed,
      child: Scaffold(
        appBar: AppBar(
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () {
              _handleBackPressed();
            },
          ),
          title: const Text(AppStrings.verifyOtp),
        ),
        body: SafeArea(
          child: LayoutBuilder(
            builder: (context, constraints) => SingleChildScrollView(
              padding: context.pagePadding,
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: constraints.maxHeight),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    SizedBox(height: context.adaptiveSpace(24)),
                    Text(
                      'Enter the 6-digit code sent to',
                      style: Theme.of(context).textTheme.titleMedium,
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: context.adaptiveSpace(8)),
                    Text(
                      widget.phoneNumber,
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: AppColors.primary,
                          ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: context.adaptiveSpace(32)),
                    LayoutBuilder(
                      builder: (context, otpConstraints) {
                        const spacing = 8.0;
                        final fieldWidth = ((otpConstraints.maxWidth - (spacing * 5)) / 6)
                            .clamp(40.0, 52.0);
                        return Wrap(
                          alignment: WrapAlignment.center,
                          spacing: spacing,
                          runSpacing: spacing,
                          children: List.generate(6, (index) {
                            return SizedBox(
                              width: fieldWidth,
                              child: TextField(
                                controller: _controllers[index],
                                focusNode: _focusNodes[index],
                                keyboardType: TextInputType.number,
                                textAlign: TextAlign.center,
                                maxLength: 1,
                                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                                style: TextStyle(
                                  fontSize: context.isCompactWidth ? 20 : 24,
                                  fontWeight: FontWeight.bold,
                                ),
                                decoration: InputDecoration(
                                  counterText: '',
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  focusedBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(8),
                                    borderSide: const BorderSide(
                                      color: AppColors.primary,
                                      width: 2,
                                    ),
                                  ),
                                ),
                                onChanged: (value) => _onOtpChanged(index, value),
                              ),
                            );
                          }),
                        );
                      },
                    ),
                    if (_errorMessage != null) ...[
                      SizedBox(height: context.adaptiveSpace(16)),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppColors.error.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _errorMessage!,
                          style: const TextStyle(color: AppColors.error),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ],
                    SizedBox(height: context.adaptiveSpace(32)),
                    CustomButton(
                      text: AppStrings.verifyOtp,
                      onPressed: _verifyOtp,
                      isLoading: _isLoading,
                    ),
                    SizedBox(height: context.adaptiveSpace(24)),
                    Wrap(
                      alignment: WrapAlignment.center,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      spacing: 4,
                      runSpacing: 4,
                      children: [
                        Text(
                          AppStrings.didntReceiveOtp,
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                        TextButton(
                          onPressed: _resendTimer > 0 || _isLoading ? null : _resendOtp,
                          child: Text(
                            _resendTimer > 0 ? '$_resendTimer s' : AppStrings.resendOtp,
                            style: TextStyle(
                              color: _resendTimer > 0 ? Colors.grey : AppColors.primary,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
