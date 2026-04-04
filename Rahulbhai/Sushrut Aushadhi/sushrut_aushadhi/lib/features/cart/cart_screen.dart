import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../core/constants/app_colors.dart';
import '../../core/utils/responsive.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../core/widgets/login_prompt_widget.dart';
import '../../core/di/service_providers.dart';
import '../../models/order_model.dart';
import '../../models/delivery_address.dart';
import '../../models/user_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/cart_provider.dart';
import '../../services/connectivity_service.dart';

class CartScreen extends ConsumerStatefulWidget {
  final bool fromProfile;

  const CartScreen({super.key, this.fromProfile = false});

  @override
  ConsumerState<CartScreen> createState() => _CartScreenState();
}

class _CartScreenState extends ConsumerState<CartScreen> {
  final _promoController = TextEditingController();
  String? _appliedPromo;
  bool _isProcessing = false;
  bool _hasAutoProcessed = false;

  @override
  void dispose() {
    _promoController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cartItems = ref.watch(cartProvider);
    final cartTotal = ref.watch(cartTotalProvider);
    final requiresPrescription = ref.watch(cartRequiresPrescriptionProvider);
    final mrpTotal = _calculateMrpTotal(cartItems);
    final discount = mrpTotal - cartTotal;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: Container(
          margin: const EdgeInsets.all(8),
          decoration: BoxDecoration(color: Colors.grey.shade200, shape: BoxShape.circle),
          child: IconButton(
            icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary, size: 20),
            onPressed: () => context.pop(),
          ),
        ),
        title: Text(
          'My Cart (${cartItems.length} items)',
          style: GoogleFonts.sora(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary,
          ),
        ),
        centerTitle: true,
      ),
      body: FirebaseAuth.instance.currentUser == null
          ? const LoginPromptWidget(
              message: 'Login to view your cart and place orders.',
            )
          : cartItems.isEmpty
              ? EmptyStateWidget(
                  emoji: '🛒',
                  title: 'Your Cart is Empty',
                  subtitle: 'Add medicines from our store to get started.',
                  buttonText: 'Browse Medicines',
                  onButtonPressed: () => context.go('/home'),
                )
              : Column(
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    padding: EdgeInsets.all(context.isCompactWidth ? 12 : 16),
                    child: Column(
                      children: [
                        if (requiresPrescription) _buildPrescriptionAlert(),
                        _buildCartItemsList(cartItems),
                        _buildPromoCodeSection(),
                        _buildBillSummary(mrpTotal, discount),
                        _trustFooter(),
                      ],
                    ),
                  ),
                ),
                _buildStickyBottomBar(cartTotal),
              ],
            ),
    );
  }

  Widget _buildPrescriptionAlert() {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.accent.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.accent),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 2),
            child: Icon(Icons.description_outlined, color: AppColors.accent, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Some items need a prescription. Please upload before placing order.',
              style: GoogleFonts.sora(fontSize: 13, color: AppColors.accent),
            ),
          ),
          IconButton(
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
            onPressed: () => context.push('/prescription?type=medicine'),
            icon: const Icon(Icons.arrow_forward_ios, color: AppColors.accent, size: 16),
          ),
        ],
      ),
    );
  }

  Widget _buildCartItemsList(List cartItems) {
    return Column(
      children: cartItems.asMap().entries.map((entry) {
        return _buildCartItemCard(entry.value);
      }).toList(),
    );
  }

  Widget _buildCartItemCard(dynamic item) {
    return Dismissible(
      key: Key(item.medicine.id),
      direction: DismissDirection.endToStart,
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        margin: const EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(
          color: AppColors.error,
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Icon(Icons.delete, color: Colors.white),
      ),
      onDismissed: (_) => ref.read(cartProvider.notifier).removeItem(item.medicine.id),
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isVerySmall = constraints.maxWidth < 340;
            if (isVerySmall) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildMedicineThumb(),
                      SizedBox(width: context.adaptiveSpace(10)),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              item.medicine.name,
                              style: GoogleFonts.sora(
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                                color: AppColors.textPrimary,
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                            const SizedBox(height: 2),
                            Text(
                              item.medicine.manufacturer,
                              style: GoogleFonts.sora(fontSize: 10, color: AppColors.textSecondary),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                            const SizedBox(height: 4),
                            FittedBox(
                              fit: BoxFit.scaleDown,
                              child: Text(
                                '\u20B9${item.medicine.price.toStringAsFixed(0)}',
                                style: GoogleFonts.sora(
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                  color: AppColors.primary,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      SizedBox(width: context.adaptiveSpace(6)),
                      GestureDetector(
                        onTap: () => ref.read(cartProvider.notifier).removeItem(item.medicine.id),
                        child: Container(
                          width: 26,
                          height: 26,
                          decoration: BoxDecoration(
                            color: const Color(0xFFFFEBEE),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(
                            Icons.delete_outline,
                            color: Color(0xFFE53935),
                            size: 14,
                          ),
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: context.adaptiveSpace(10)),
                  Align(
                    alignment: Alignment.centerRight,
                    child: _buildQuantityControls(item, isVerySmall: true),
                  ),
                ],
              );
            }

            return Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildMedicineThumb(),
                SizedBox(width: context.adaptiveSpace(10)),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.medicine.name,
                        style: GoogleFonts.sora(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: AppColors.textPrimary,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        item.medicine.manufacturer,
                        style: GoogleFonts.sora(fontSize: 10, color: AppColors.textSecondary),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      FittedBox(
                        fit: BoxFit.scaleDown,
                        child: Text(
                          '\u20B9${item.medicine.price.toStringAsFixed(0)}',
                          style: GoogleFonts.sora(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: AppColors.primary,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(width: context.adaptiveSpace(8)),
                SizedBox(
                  width: 84.0,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      GestureDetector(
                        onTap: () => ref.read(cartProvider.notifier).removeItem(item.medicine.id),
                        child: Container(
                          width: 26,
                          height: 26,
                          decoration: BoxDecoration(
                            color: const Color(0xFFFFEBEE),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(
                            Icons.delete_outline,
                            color: Color(0xFFE53935),
                            size: 14,
                          ),
                        ),
                      ),
                      SizedBox(height: context.adaptiveSpace(6)),
                      _buildQuantityControls(item, isVerySmall: false),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildQuantityControls(dynamic item, {required bool isVerySmall}) {
    return Container(
      height: 28,
      decoration: BoxDecoration(
        color: const Color(0xFFF5F7F5),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFEDF2ED)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          GestureDetector(
            onTap: () {
              if (item.quantity > 1) {
                ref.read(cartProvider.notifier).updateQuantity(
                      item.medicine.id,
                      item.quantity - 1,
                    );
              } else {
                ref.read(cartProvider.notifier).removeItem(item.medicine.id);
              }
            },
            child: SizedBox(
              width: isVerySmall ? 22.0 : 26.0,
              height: 28,
              child: Center(
                child: Text(
                  '−',
                  style: GoogleFonts.sora(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.primary,
                  ),
                ),
              ),
            ),
          ),
          Container(
            width: isVerySmall ? 24.0 : 28.0,
            height: 28,
            alignment: Alignment.center,
            child: FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                '${item.quantity}',
                style: GoogleFonts.sora(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary,
                ),
              ),
            ),
          ),
          GestureDetector(
            onTap: () => ref.read(cartProvider.notifier).updateQuantity(
                  item.medicine.id,
                  item.quantity + 1,
                ),
            child: SizedBox(
              width: isVerySmall ? 22.0 : 26.0,
              height: 28,
              child: Center(
                child: Text(
                  '+',
                  style: GoogleFonts.sora(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.primary,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMedicineThumb() {
    return Container(
      width: 56,
      height: 56,
      decoration: BoxDecoration(
        color: AppColors.primaryLight,
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Center(
        child: Icon(Icons.medication_rounded, size: 28, color: AppColors.primary),
      ),
    );
  }

  Widget _buildPromoCodeSection() {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Promo Code',
            style: GoogleFonts.sora(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 12),
          LayoutBuilder(
            builder: (context, constraints) {
              final compact = constraints.maxWidth < 360;
              final applyWidget = _appliedPromo != null
                  ? Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.check_circle, color: AppColors.primary, size: 18),
                          const SizedBox(width: 4),
                          Text(
                            _appliedPromo!,
                            style: GoogleFonts.sora(
                              color: AppColors.primary,
                              fontWeight: FontWeight.w600,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    )
                  : TextButton(
                      onPressed: () {
                        if (_promoController.text.isNotEmpty) {
                          setState(() => _appliedPromo = _promoController.text);
                        }
                      },
                      child: Text(
                        'APPLY',
                        style: GoogleFonts.sora(
                          color: AppColors.primary,
                          fontWeight: FontWeight.w600,
                          fontSize: 14,
                        ),
                      ),
                    );

              if (compact) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildPromoField(),
                    const SizedBox(height: 12),
                    Align(alignment: Alignment.centerLeft, child: applyWidget),
                  ],
                );
              }

              return Row(
                children: [
                  Expanded(child: _buildPromoField()),
                  const SizedBox(width: 12),
                  applyWidget,
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildPromoField() {
    return TextField(
      controller: _promoController,
      style: GoogleFonts.sora(fontSize: 14),
      decoration: InputDecoration(
        hintText: 'Enter promo code',
        hintStyle: GoogleFonts.sora(fontSize: 14, color: AppColors.textSecondary),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      ),
      onChanged: (_) => setState(() {}),
    );
  }

  Widget _buildBillSummary(double mrpTotal, double discount) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Bill Summary',
            style: GoogleFonts.sora(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 12),
          _buildSummaryRow('MRP Total', '\u20B9${mrpTotal.toStringAsFixed(0)}'),
          if (discount > 0)
            _buildSummaryRow('Discount', '-\u20B9${discount.toStringAsFixed(0)}', isGreen: true),
          _buildSummaryRow('Delivery', 'FREE', isGreen: true),
          const Divider(height: 24),
          _buildSummaryRow(
            'Total Payable',
            '\u20B9${(mrpTotal - discount).toStringAsFixed(0)}',
            isBold: true,
            isGreen: true,
          ),
          if (discount > 0) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.celebration_outlined, size: 14, color: AppColors.primary),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      'You save \u20B9${discount.toStringAsFixed(0)} on this order!',
                      style: GoogleFonts.sora(
                        color: AppColors.primary,
                        fontWeight: FontWeight.w600,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSummaryRow(String label, String value, {bool isBold = false, bool isGreen = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: GoogleFonts.sora(
                fontSize: 13,
                fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
                color: AppColors.textPrimary,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            value,
            style: GoogleFonts.sora(
              fontSize: 13,
              fontWeight: isBold ? FontWeight.bold : FontWeight.w600,
              color: isGreen ? AppColors.primary : AppColors.textPrimary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStickyBottomBar(double cartTotal) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(28),
          topRight: Radius.circular(28),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade300,
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _secureCheckoutBadge(),
            const SizedBox(height: 10),
            LayoutBuilder(
          builder: (context, constraints) {
            final compact = constraints.maxWidth < 360;
            final priceInfo = Column(
              crossAxisAlignment: compact ? CrossAxisAlignment.center : CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Total Payable',
                  style: GoogleFonts.sora(fontSize: 12, color: AppColors.textSecondary),
                ),
                FittedBox(
                  fit: BoxFit.scaleDown,
                  child: Text(
                    '\u20B9${cartTotal.toStringAsFixed(0)}',
                    style: GoogleFonts.sora(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: AppColors.primary,
                    ),
                  ),
                ),
              ],
            );

            final action = SizedBox(
              width: compact ? double.infinity : null,
              child: ElevatedButton(
                onPressed: _isProcessing ? null : _placeOrder,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  elevation: 0,
                ),
                child: _isProcessing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                      )
                    : Text(
                        'Place Order',
                        style: GoogleFonts.sora(fontWeight: FontWeight.w600),
                      ),
              ),
            );

            if (compact) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  priceInfo,
                  const SizedBox(height: 12),
                  action,
                ],
              );
            }

            return Row(
              children: [
                Expanded(child: priceInfo),
                const SizedBox(width: 12),
                action,
              ],
            );
          },
        ),
          ],
        ),
      ),
    );
  }

  Widget _secureCheckoutBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFE1F5EE),
        borderRadius: BorderRadius.circular(8),
      ),
      child: const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text('🔒', style: TextStyle(fontSize: 13)),
          SizedBox(width: 6),
          Text(
            'Safe & Secure Checkout  •  Cash on Delivery',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: Color(0xFF0F6E56),
            ),
          ),
        ],
      ),
    );
  }

  Widget _trustFooter() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _footerTrust('✅', 'Genuine'),
          const SizedBox(width: 16),
          _footerTrust('🚚', 'Fast Delivery'),
          const SizedBox(width: 16),
          _footerTrust('↩️', 'Easy Returns'),
        ],
      ),
    );
  }

  Widget _footerTrust(String emoji, String label) {
    return Row(
      children: [
        Text(emoji, style: const TextStyle(fontSize: 12)),
        const SizedBox(width: 4),
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            color: AppColors.textSecondary,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  double _calculateMrpTotal(List cartItems) {
    double total = 0;
    for (final item in cartItems) {
      total += item.medicine.mrp * item.quantity;
    }
    return total;
  }

  Future<void> _placeOrder() async {
    final isOnline = await ConnectivityService.checkConnection();
    if (!isOnline) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              const Icon(Icons.wifi_off, color: Colors.white, size: 16),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'No internet connection.\nPlease connect to place your order.',
                  style: GoogleFonts.sora(),
                ),
              ),
            ],
          ),
          backgroundColor: const Color(0xFFE53935),
          duration: const Duration(seconds: 4),
        ),
      );
      return;
    }

    final user = FirebaseAuth.instance.currentUser;
    final requiresPrescription = ref.read(cartRequiresPrescriptionProvider);

    if (user == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please login to place your order', style: GoogleFonts.sora()),
          backgroundColor: AppColors.primary,
        ),
      );
      context.push('/login');
      return;
    }

    // Await the latest settled value from the stream — ensures we never validate
    // against a stale snapshot (e.g. after returning from the profile screen).
    final UserModel? currentUser = await ref
        .read(currentUserProvider.future)
        .catchError((_) => null);
    if (currentUser == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please login to place your order', style: GoogleFonts.sora()),
          backgroundColor: AppColors.primary,
        ),
      );
      context.push('/login');
      return;
    }

    // If returning from profile after saving address, auto-proceed without
    // re-prompting — data is now fresh so _createOrder will use valid address.
    if (widget.fromProfile && !_hasAutoProcessed) {
      _hasAutoProcessed = true;
      await _createOrder();
      return;
    }

    final hasPhone = currentUser.phone.isNotEmpty;
    final hasAddressString = currentUser.address.isNotEmpty;
    final hasDeliveryAddress = currentUser.deliveryAddress.line1.isNotEmpty ||
        currentUser.deliveryAddress.city.isNotEmpty ||
        currentUser.deliveryAddress.pincode.isNotEmpty;
    final hasAddress = hasAddressString || hasDeliveryAddress;
    final hasPincode = currentUser.pincode.isNotEmpty ||
        currentUser.deliveryAddress.pincode.isNotEmpty;

    if (!hasPhone || !hasAddress || !hasPincode) {
      final shouldFillAddress = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('Complete Your Profile', style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
          content: Text(
            'Please fill in your phone number, address, and pincode to place an order.',
            style: GoogleFonts.sora(),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: Text('Cancel', style: GoogleFonts.sora(color: AppColors.textSecondary)),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: Text('Fill Address', style: GoogleFonts.sora(color: AppColors.primary)),
            ),
          ],
        ),
      );

      if (shouldFillAddress == true && mounted) {
        context.go('/profile?redirectTo=cart');
      }
      return;
    }

    if (requiresPrescription) {
      context.push('/prescription?type=medicine');
      return;
    }

    await _createOrder();
  }

  Future<void> _createOrder() async {
    setState(() => _isProcessing = true);

    try {
      final cartItems = ref.read(cartProvider);
      final cartTotal = ref.read(cartTotalProvider);
      final currentUserAsync = ref.read(currentUserProvider);
      final authStateAsync = ref.read(authStateProvider);

      final user = currentUserAsync.valueOrNull;
      final authUser = authStateAsync.valueOrNull;

      if (user == null || authUser == null) {
        throw Exception('User not authenticated');
      }

      final orderItemsList = cartItems.map((item) {
        return OrderItem(
          medicineId: item.medicine.id,
          medicineName: item.medicine.name,
          price: item.medicine.price,
          quantity: item.quantity,
          subtotal: item.subtotal,
        );
      }).toList();

      final deliveryAddress = user.deliveryAddress.line1.isNotEmpty
          ? user.deliveryAddress
          : DeliveryAddress(
              line1: user.address,
              city: '',
              state: '',
              pincode: user.pincode,
            );

      final order = OrderModel(
        orderId: '',
        userId: authUser.uid,
        userPhone: user.phone,
        userName: user.name.isNotEmpty ? user.name : 'Customer',
        deliveryAddress: deliveryAddress,
        items: orderItemsList,
        totalAmount: cartTotal,
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
      );

      final newOrderId = await ref.read(firestoreServiceProvider).placeOrder(order);
      
      final shortOrderId = newOrderId.length > 6
          ? newOrderId.substring(newOrderId.length - 6).toUpperCase()
          : newOrderId.toUpperCase();

      if (!mounted) {
        return;
      }

      final itemsData = cartItems.map((item) => {
        'medicineId': item.medicine.id,
        'medicineName': item.medicine.name,
        'price': item.medicine.price,
        'quantity': item.quantity,
        'subtotal': item.subtotal,
      }).toList();

      final deliveryAddressString = user.deliveryAddress.line1.isNotEmpty
          ? user.deliveryAddress.toDisplayString()
          : user.address.isNotEmpty
              ? user.address
              : 'Address not provided';

      ref.read(cartProvider.notifier).clearCart();

      if (!mounted) {
        return;
      }

      context.go('/order-confirmation', extra: {
        'orderId': newOrderId,
        'totalAmount': cartTotal,
        'items': itemsData,
        'deliveryAddress': deliveryAddressString,
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: AppColors.error),
      );
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }
}
