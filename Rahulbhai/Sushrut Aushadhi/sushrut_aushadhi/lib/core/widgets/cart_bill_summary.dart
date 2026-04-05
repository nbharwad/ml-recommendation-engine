import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

class CartBillSummary extends StatelessWidget {
  final double mrpTotal;
  final double cartTotal;
  final double discount;
  final double deliveryCharges;
  final double finalTotal;

  const CartBillSummary({
    super.key,
    required this.mrpTotal,
    required this.cartTotal,
    required this.discount,
    this.deliveryCharges = 0,
    required this.finalTotal,
  });

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFE8ECE7)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.receipt_long_rounded,
                    color: AppColors.primary, size: 20),
                const SizedBox(width: 8),
                Text(
                  'Bill Summary',
                  style: GoogleFonts.sora(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildRow('Total MRP', mrpTotal),
            _buildRow('Discount', -discount, isDiscount: true),
            if (deliveryCharges > 0)
              _buildRow('Delivery Charges', deliveryCharges),
            const Divider(height: 20),
            _buildRow('Total', finalTotal, isTotal: true),
          ],
        ),
      ),
    );
  }

  Widget _buildRow(String label, double amount,
      {bool isDiscount = false, bool isTotal = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: GoogleFonts.sora(
              fontSize: 13,
              color: isTotal ? AppColors.textPrimary : AppColors.textSecondary,
              fontWeight: isTotal ? FontWeight.w700 : FontWeight.w500,
            ),
          ),
          Text(
            amount == 0 ? 'Free' : '\u20B9${amount.abs().toStringAsFixed(0)}',
            style: GoogleFonts.sora(
              fontSize: isTotal ? 16 : 13,
              color: isDiscount
                  ? AppColors.statusDelivered
                  : isTotal
                      ? AppColors.primary
                      : AppColors.textPrimary,
              fontWeight: isTotal ? FontWeight.w800 : FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
