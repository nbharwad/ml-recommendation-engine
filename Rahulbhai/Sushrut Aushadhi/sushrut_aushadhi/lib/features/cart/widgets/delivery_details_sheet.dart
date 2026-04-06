import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../core/constants/app_colors.dart';
import '../../../services/delivery_details_service.dart';

class DeliveryDetailsSheet extends StatefulWidget {
  final DeliveryDetails? existingDetails;
  final Future<void> Function(DeliveryDetails) onSaved;

  const DeliveryDetailsSheet({
    super.key,
    this.existingDetails,
    required this.onSaved,
  });

  @override
  State<DeliveryDetailsSheet> createState() => _DeliveryDetailsSheetState();
}

class _DeliveryDetailsSheetState extends State<DeliveryDetailsSheet> {
  final _phoneCtrl = TextEditingController();
  final _addressCtrl = TextEditingController();
  final _pincodeCtrl = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  bool _isSaving = false;
  String? _phoneError;
  String? _addressError;
  String? _pincodeError;

  @override
  void initState() {
    super.initState();
    _prefillData();
  }

  void _prefillData() {
    final existing = widget.existingDetails;
    final user = FirebaseAuth.instance.currentUser;

    if (user?.phoneNumber?.isNotEmpty == true) {
      _phoneCtrl.text = user!.phoneNumber!;
    } else if (existing?.phone.isNotEmpty == true) {
      _phoneCtrl.text = existing!.phone;
    }

    if (existing?.address.isNotEmpty == true) {
      _addressCtrl.text = existing!.address;
    }
    if (existing?.pincode.isNotEmpty == true) {
      _pincodeCtrl.text = existing!.pincode;
    }
  }

  @override
  void dispose() {
    _phoneCtrl.dispose();
    _addressCtrl.dispose();
    _pincodeCtrl.dispose();
    super.dispose();
  }

  bool _validate() {
    bool valid = true;
    setState(() {
      _phoneError = null;
      _addressError = null;
      _pincodeError = null;

      final phone = _phoneCtrl.text.trim();
      if (phone.isEmpty) {
        _phoneError = 'Phone number is required';
        valid = false;
      } else if (phone.replaceAll(RegExp(r'[+\s]'), '').length < 10) {
        _phoneError = 'Enter valid 10-digit number';
        valid = false;
      }

      final address = _addressCtrl.text.trim();
      if (address.isEmpty) {
        _addressError = 'Delivery address is required';
        valid = false;
      } else if (address.length < 5) {
        _addressError = 'Enter complete address';
        valid = false;
      }

      final pincode = _pincodeCtrl.text.trim();
      if (pincode.isEmpty) {
        _pincodeError = 'Pincode is required';
        valid = false;
      } else if (pincode.length != 6 || !RegExp(r'^\d{6}$').hasMatch(pincode)) {
        _pincodeError = 'Enter valid 6-digit pincode';
        valid = false;
      }
    });
    return valid;
  }

  Future<void> _save() async {
    if (!_validate()) return;

    setState(() => _isSaving = true);

    try {
      final phone = _phoneCtrl.text.trim();
      final address = _addressCtrl.text.trim();
      final pincode = _pincodeCtrl.text.trim();

      await DeliveryDetailsService.saveDetails(
        phone: phone,
        address: address,
        pincode: pincode,
      );

      final details = DeliveryDetails(
        phone: phone,
        address: address,
        pincode: pincode,
      );

      if (!mounted) return;

      Navigator.of(context).pop();
      await widget.onSaved(details);
    } catch (e) {
      debugPrint('Save error: $e');
      if (mounted) {
        setState(() => _isSaving = false);
        String msg = e.toString().replaceAll('Exception:', '').trim();
        if (msg.contains('permission')) {
          msg = 'Permission denied. Please logout and login again.';
        } else if (msg.contains('cloud_firestore')) {
          msg = 'Connection error. Check internet and try again.';
        } else if (msg.contains('Not logged in')) {
          msg = 'Session expired. Please login again.';
        }
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(msg, style: GoogleFonts.sora()),
            backgroundColor: AppColors.error,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;
    final bottomPadding = MediaQuery.of(context).padding.bottom;
    final authPhone = FirebaseAuth.instance.currentUser?.phoneNumber;
    final hasAuthPhone = authPhone?.isNotEmpty == true;

    return Padding(
      padding: EdgeInsets.only(bottom: bottomInset),
      child: Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: SafeArea(
          top: false,
          child: SingleChildScrollView(
            padding: EdgeInsets.fromLTRB(20, 12, 20, 16 + bottomPadding),
            child: Form(
              key: _formKey,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 36,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade300,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: const Color(0xFFE1F5EE),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Center(
                        child: Text('📍', style: TextStyle(fontSize: 22)),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Delivery Details',
                          style: GoogleFonts.sora(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          'Required to place your order',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
                  ]),
                  const SizedBox(height: 24),
                  _buildField(
                    label: 'Phone Number',
                    controller: _phoneCtrl,
                    icon: Icons.phone_outlined,
                    hint: '+91XXXXXXXXXX',
                    keyboardType: TextInputType.phone,
                    errorText: _phoneError,
                    readOnly: hasAuthPhone,
                    helperText:
                        hasAuthPhone ? 'Auto-filled from your login' : null,
                  ),
                  const SizedBox(height: 14),
                  _buildField(
                    label: 'Delivery Address',
                    controller: _addressCtrl,
                    icon: Icons.home_outlined,
                    hint: 'House no, Street, Area',
                    maxLines: 2,
                    errorText: _addressError,
                  ),
                  const SizedBox(height: 14),
                  _buildField(
                    label: 'Pincode',
                    controller: _pincodeCtrl,
                    icon: Icons.location_on_outlined,
                    hint: '6-digit pincode',
                    keyboardType: TextInputType.number,
                    maxLength: 6,
                    errorText: _pincodeError,
                  ),
                  const SizedBox(height: 24),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _isSaving ? null : _save,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        disabledBackgroundColor:
                            AppColors.primary.withOpacity(0.6),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                        elevation: 0,
                      ),
                      child: _isSaving
                          ? Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Text(
                                  'Saving...',
                                  style: GoogleFonts.sora(
                                    color: Colors.white,
                                    fontSize: 15,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            )
                          : Text(
                              'Save & Continue',
                              style: GoogleFonts.sora(
                                color: Colors.white,
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField({
    required String label,
    required TextEditingController controller,
    required IconData icon,
    required String hint,
    TextInputType keyboardType = TextInputType.text,
    int maxLines = 1,
    int? maxLength,
    String? errorText,
    bool readOnly = false,
    String? helperText,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextField(
          controller: controller,
          keyboardType: keyboardType,
          maxLines: maxLines,
          maxLength: maxLength,
          readOnly: readOnly,
          onChanged: (_) {
            if (errorText != null) {
              setState(() {
                if (label.contains('Phone')) _phoneError = null;
                if (label.contains('Address')) _addressError = null;
                if (label.contains('Pin')) _pincodeError = null;
              });
            }
          },
          style: TextStyle(
            fontSize: 14,
            color: readOnly ? Colors.grey.shade600 : Colors.black,
          ),
          decoration: InputDecoration(
            labelText: label,
            hintText: hint,
            hintStyle: TextStyle(
              fontSize: 13,
              color: Colors.grey.shade400,
            ),
            prefixIcon: Icon(
              icon,
              color: errorText != null ? Colors.red : AppColors.primary,
              size: 20,
            ),
            filled: true,
            fillColor: readOnly ? Colors.grey.shade50 : Colors.grey.shade50,
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide(
                color: errorText != null
                    ? Colors.red.shade300
                    : Colors.grey.shade200,
              ),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide(
                color: errorText != null ? Colors.red : AppColors.primary,
                width: 1.5,
              ),
            ),
            errorText: errorText,
            helperText: helperText,
            helperStyle: TextStyle(
              fontSize: 10,
              color: AppColors.primary,
            ),
            counterText: '',
          ),
        ),
      ],
    );
  }
}
