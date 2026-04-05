import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class HomeOfferStrip extends StatelessWidget {
  const HomeOfferStrip({super.key});

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isCompact = constraints.maxWidth < 360;

            return Container(
              padding: EdgeInsets.symmetric(
                horizontal: isCompact ? 12 : 16,
                vertical: isCompact ? 12 : 14,
              ),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF3D5),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFF3C158)),
              ),
              child: isCompact ? _buildCompactLayout() : _buildNormalLayout(),
            );
          },
        ),
      ),
    );
  }

  Widget _buildCompactLayout() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.local_offer_rounded,
                size: 20, color: Color(0xFFFF8A00)),
            const SizedBox(width: 8),
            Expanded(child: _buildOfferText(isCompact: true)),
          ],
        ),
        const SizedBox(height: 10),
        _buildOfferCode(isCompact: true),
      ],
    );
  }

  Widget _buildNormalLayout() {
    return Row(
      children: [
        const Icon(Icons.local_offer_rounded,
            size: 24, color: Color(0xFFFF8A00)),
        const SizedBox(width: 12),
        Expanded(child: _buildOfferText()),
        const SizedBox(width: 12),
        _buildOfferCode(),
      ],
    );
  }

  Widget _buildOfferText({bool isCompact = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '20% off on your first order!',
          style: GoogleFonts.sora(
            color: const Color(0xFFB96A00),
            fontSize: isCompact ? 13 : 14,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          'Apply the code at checkout for instant savings.',
          style: GoogleFonts.sora(
            color: const Color(0xFFB58228),
            fontSize: isCompact ? 10 : 11,
          ),
        ),
      ],
    );
  }

  Widget _buildOfferCode({bool isCompact = false}) {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: isCompact ? 10 : 14,
        vertical: isCompact ? 6 : 8,
      ),
      decoration: BoxDecoration(
        color: const Color(0xFFFF8A00),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'SUSHRUT20',
        style: GoogleFonts.sora(
          color: Colors.white,
          fontSize: isCompact ? 10 : 12,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
