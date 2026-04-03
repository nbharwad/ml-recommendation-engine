import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_strings.dart';
import '../../core/utils/responsive.dart';
import '../../core/utils/validators.dart';
import '../../core/widgets/custom_button.dart';
import '../../core/widgets/custom_text_field.dart';
import '../../core/di/service_providers.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _phoneController = TextEditingController();
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void dispose() {
    _phoneController.dispose();
    super.dispose();
  }

  Future<void> _sendOtp() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    final phone = '+91${_phoneController.text.trim()}';

    await ref.read(authServiceProvider).sendOtp(
      phoneNumber: phone,
      onCodeSent: (verificationId) {
        if (!mounted) {
          return;
        }
        setState(() => _isLoading = false);
        context.push('/otp', extra: {
          'verificationId': verificationId,
          'phoneNumber': phone,
        });
      },
      onVerificationCompleted: (_) {
        if (!mounted) {
          return;
        }
        setState(() => _isLoading = false);
        context.go('/home');
      },
      onTimeout: (_) {
        if (!mounted) {
          return;
        }
        setState(() => _isLoading = false);
      },
      onError: (error) {
        if (!mounted) {
          return;
        }
        setState(() {
          _errorMessage = error;
          _isLoading = false;
        });
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final iconBoxSize = context.isCompactWidth ? 84.0 : 100.0;
    final iconSize = context.isCompactWidth ? 48.0 : 60.0;

    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) => SingleChildScrollView(
            padding: context.pagePadding,
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    SizedBox(height: context.isShortHeight ? 16 : 40),
                    Center(
                      child: Container(
                        width: iconBoxSize,
                        height: iconBoxSize,
                        decoration: BoxDecoration(
                          color: AppColors.primaryLight,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Icon(
                          Icons.local_pharmacy,
                          size: iconSize,
                          color: AppColors.primary,
                        ),
                      ),
                    ),
                    SizedBox(height: context.adaptiveSpace(32)),
                    Text(
                      AppStrings.appName,
                      style: Theme.of(context).textTheme.headlineMedium,
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: context.adaptiveSpace(8)),
                    Text(
                      'Your trusted medicine partner',
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: context.adaptiveSpace(48)),
                    Text(
                      AppStrings.enterPhone,
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    SizedBox(height: context.adaptiveSpace(12)),
                    CustomTextField(
                      controller: _phoneController,
                      hintText: 'Enter 10-digit phone number',
                      keyboardType: TextInputType.phone,
                      prefixIcon: Icons.phone,
                      validator: Validators.validatePhone,
                      inputFormatters: [
                        FilteringTextInputFormatter.digitsOnly,
                        LengthLimitingTextInputFormatter(10),
                      ],
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
                      text: AppStrings.sendOtp,
                      onPressed: _sendOtp,
                      isLoading: _isLoading,
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
