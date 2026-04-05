import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/constants/app_colors.dart';
import '../../core/utils/responsive.dart';
import '../../providers/auth_provider.dart';

class HomeHeaderWidget extends ConsumerWidget {
  final VoidCallback onSearchTap;
  final VoidCallback onLocationTap;
  final String? currentAddress;

  const HomeHeaderWidget({
    super.key,
    required this.onSearchTap,
    required this.onLocationTap,
    this.currentAddress,
  });

  String _extractCity(String address) {
    if (address.isEmpty) return 'Select Location';
    final parts = address.split(',');
    if (parts.length >= 2) return parts[1].trim();
    return parts.first.trim();
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userAsync = ref.watch(currentUserProvider);
    final user = userAsync.valueOrNull;
    final displayAddress = user?.address ?? currentAddress ?? '';
    final city = _extractCity(displayAddress);

    return SliverAppBar(
      expandedHeight: 0,
      pinned: true,
      floating: true,
      snap: true,
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.white,
      elevation: 0,
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(120),
        child: Container(
          color: Colors.white,
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Column(
            children: [
              _buildLocationRow(city, displayAddress),
              const SizedBox(height: 12),
              _buildSearchBar(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLocationRow(String city, String displayAddress) {
    return GestureDetector(
      onTap: onLocationTap,
      child: Row(
        children: [
          const Icon(
            Icons.location_on_rounded,
            color: AppColors.primary,
            size: 20,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Deliver to',
                  style: GoogleFonts.sora(
                    fontSize: 11,
                    color: AppColors.textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Text(
                  city.isNotEmpty ? city : 'Anand, Gujarat',
                  style: GoogleFonts.sora(
                    fontSize: 15,
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
          const Icon(
            Icons.keyboard_arrow_down_rounded,
            color: AppColors.textSecondary,
            size: 20,
          ),
        ],
      ),
    );
  }

  Widget _buildSearchBar(BuildContext context) {
    return GestureDetector(
      onTap: onSearchTap,
      child: Container(
        height: 52,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        decoration: BoxDecoration(
          color: const Color(0xFFF4F5F3),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFEDEFEA)),
        ),
        child: Row(
          children: [
            const Icon(
              Icons.search_rounded,
              color: Color(0xFF98A09B),
              size: 22,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Search medicines, health products...',
                style: GoogleFonts.sora(
                  fontSize: 13,
                  color: const Color(0xFF98A09B),
                ),
              ),
            ),
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: AppColors.primary,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.mic_none_rounded,
                color: Colors.white,
                size: 20,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class HomeHeaderGuest extends StatelessWidget {
  final VoidCallback onSearchTap;

  const HomeHeaderGuest({
    super.key,
    required this.onSearchTap,
  });

  @override
  Widget build(BuildContext context) {
    return SliverAppBar(
      expandedHeight: 0,
      pinned: true,
      floating: true,
      snap: true,
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.white,
      elevation: 0,
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(120),
        child: Container(
          color: Colors.white,
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Column(
            children: [
              _buildLocationRow(),
              const SizedBox(height: 12),
              _buildSearchBar(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLocationRow() {
    return Row(
      children: [
        const Icon(
          Icons.location_on_rounded,
          color: AppColors.primary,
          size: 20,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Deliver to',
                style: GoogleFonts.sora(
                  fontSize: 11,
                  color: AppColors.textSecondary,
                  fontWeight: FontWeight.w500,
                ),
              ),
              Text(
                'Anand, Gujarat',
                style: GoogleFonts.sora(
                  fontSize: 15,
                  color: AppColors.textPrimary,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),
        const Icon(
          Icons.keyboard_arrow_down_rounded,
          color: AppColors.textSecondary,
          size: 20,
        ),
      ],
    );
  }

  Widget _buildSearchBar(BuildContext context) {
    return GestureDetector(
      onTap: onSearchTap,
      child: Container(
        height: 52,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        decoration: BoxDecoration(
          color: const Color(0xFFF4F5F3),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFEDEFEA)),
        ),
        child: Row(
          children: [
            const Icon(
              Icons.search_rounded,
              color: Color(0xFF98A09B),
              size: 22,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Search medicines, health products...',
                style: GoogleFonts.sora(
                  fontSize: 13,
                  color: const Color(0xFF98A09B),
                ),
              ),
            ),
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: AppColors.primary,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.mic_none_rounded,
                color: Colors.white,
                size: 20,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
