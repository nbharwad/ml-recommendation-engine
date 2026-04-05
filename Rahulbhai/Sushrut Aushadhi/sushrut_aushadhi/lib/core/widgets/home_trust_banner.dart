import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

class HomeTrustBanner extends StatelessWidget {
  const HomeTrustBanner({super.key});

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: Container(
        color: AppColors.primaryLight,
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          physics: const BouncingScrollPhysics(),
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Row(
            children: [
              _TrustBadge(
                icon: Icons.verified_rounded,
                label: 'Licensed',
              ),
              _buildDivider(),
              _TrustBadge(
                icon: Icons.medical_services_rounded,
                label: 'Pharmacist',
              ),
              _buildDivider(),
              _TrustBadge(
                icon: Icons.local_shipping_rounded,
                label: '2Hr Delivery',
              ),
              _buildDivider(),
              _TrustBadge(
                icon: Icons.inventory_2_rounded,
                label: '19K+ Meds',
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDivider() {
    return Container(
      width: 0.5,
      height: 28,
      margin: const EdgeInsets.symmetric(horizontal: 12),
      color: const Color(0xFF9FE1CB),
    );
  }
}

class _TrustBadge extends StatelessWidget {
  final IconData icon;
  final String label;

  const _TrustBadge({
    required this.icon,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: AppColors.primary),
          const SizedBox(width: 6),
          Text(
            label,
            style: GoogleFonts.sora(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: AppColors.primary,
            ),
          ),
        ],
      ),
    );
  }
}

class StoreClosedBanner extends StatelessWidget {
  final String storeOpenTime;

  const StoreClosedBanner({
    super.key,
    required this.storeOpenTime,
  });

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
        color: const Color(0xFFFFEBEE),
        child: Row(
          children: [
            const Icon(Icons.access_time, color: Colors.red, size: 16),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'Store is currently closed. Opens at $storeOpenTime. You can still browse!',
                style: GoogleFonts.sora(
                  fontSize: 12,
                  color: Colors.red.shade700,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
