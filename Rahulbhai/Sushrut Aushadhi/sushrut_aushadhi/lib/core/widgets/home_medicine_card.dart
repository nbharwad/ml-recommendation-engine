import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/constants/app_colors.dart';
import '../../core/utils/helpers.dart';

class HomeMedicineCard extends StatelessWidget {
  final Map<String, dynamic> medicine;
  final VoidCallback onTap;
  final VoidCallback onAddToCart;

  const HomeMedicineCard({
    super.key,
    required this.medicine,
    required this.onTap,
    required this.onAddToCart,
  });

  IconData _getMedicineIcon(String category) {
    switch (category) {
      case 'fever':
        return Icons.thermostat_rounded;
      case 'pain':
        return Icons.medication_rounded;
      case 'skin':
        return Icons.spa_rounded;
      case 'diabetes':
        return Icons.bloodtype_rounded;
      case 'heart':
        return Icons.favorite_rounded;
      case 'vitamins':
        return Icons.health_and_safety_rounded;
      default:
        return Icons.medication_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final price = ((medicine['price'] ?? 0) as num).toDouble();
    final mrp = ((medicine['mrp'] ?? 0) as num).toDouble();
    final discount = Helpers.calculateDiscount(price, mrp);
    final requiresPrescription = medicine['requiresPrescription'] == true;
    final category = medicine['category']?.toString() ?? '';

    return LayoutBuilder(
      builder: (context, constraints) {
        final isCompact = constraints.maxWidth < 170;
        final imageSize = constraints.maxWidth * (isCompact ? 0.42 : 0.36);
        final spacing = isCompact ? 8.0 : 10.0;

        return GestureDetector(
          onTap: onTap,
          child: Container(
            padding: EdgeInsets.all(isCompact ? 10 : 12),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: const Color(0xFFE8ECE7)),
              boxShadow: const [
                BoxShadow(
                  color: Color(0x0A122019),
                  blurRadius: 14,
                  offset: Offset(0, 8),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    if (requiresPrescription)
                      Container(
                        padding: EdgeInsets.symmetric(
                          horizontal: isCompact ? 5 : 6,
                          vertical: isCompact ? 2 : 3,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFEBEE),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          'Rx',
                          style: GoogleFonts.sora(
                            color: AppColors.error,
                            fontSize: isCompact ? 8 : 9,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      )
                    else
                      const SizedBox.shrink(),
                    const Spacer(),
                    GestureDetector(
                      onTap: onAddToCart,
                      child: Container(
                        width: isCompact ? 26 : 28,
                        height: isCompact ? 26 : 28,
                        decoration: BoxDecoration(
                          color: AppColors.primaryLight,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Icon(
                          Icons.add_shopping_cart_rounded,
                          size: isCompact ? 14 : 15,
                          color: AppColors.primary,
                        ),
                      ),
                    ),
                  ],
                ),
                SizedBox(height: spacing),
                Center(
                  child: Container(
                    width: imageSize,
                    height: imageSize,
                    decoration: BoxDecoration(
                      color: AppColors.primaryLight,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Center(
                      child: Icon(
                        _getMedicineIcon(category),
                        size: isCompact ? 30 : 36,
                        color: AppColors.primary,
                      ),
                    ),
                  ),
                ),
                SizedBox(height: spacing),
                Expanded(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Flexible(
                        fit: FlexFit.loose,
                        child: Text(
                          medicine['name']?.toString() ?? '',
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.sora(
                            fontSize: isCompact ? 12 : 13,
                            height: 1.3,
                            color: AppColors.textPrimary,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      SizedBox(height: isCompact ? 3 : 4),
                      Flexible(
                        fit: FlexFit.loose,
                        child: Text(
                          medicine['brand']?.toString() ?? '',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.sora(
                            fontSize: isCompact ? 10 : 11,
                            color: AppColors.textSecondary,
                          ),
                        ),
                      ),
                      const Spacer(),
                      if (discount > 0)
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          margin: const EdgeInsets.only(bottom: 4),
                          decoration: BoxDecoration(
                            color: AppColors.error.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            '$discount% off',
                            style: GoogleFonts.sora(
                              color: AppColors.error,
                              fontSize: 9,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      Row(
                        children: [
                          Text(
                            '\u20B9${price.toStringAsFixed(0)}',
                            style: GoogleFonts.sora(
                              color: AppColors.primary,
                              fontSize: isCompact ? 13 : 14,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                          const SizedBox(width: 4),
                          if (mrp > price)
                            Text(
                              '\u20B9${mrp.toStringAsFixed(0)}',
                              style: GoogleFonts.sora(
                                color: AppColors.textSecondary,
                                fontSize: isCompact ? 9 : 10,
                                decoration: TextDecoration.lineThrough,
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class HomeMedicineGrid extends StatelessWidget {
  final List<Map<String, dynamic>> medicines;
  final Function(Map<String, dynamic>) onMedicineTap;
  final Function(Map<String, dynamic>) onAddToCart;

  const HomeMedicineGrid({
    super.key,
    required this.medicines,
    required this.onMedicineTap,
    required this.onAddToCart,
  });

  @override
  Widget build(BuildContext context) {
    final displayMedicines = medicines.take(6).toList();

    return SliverToBoxAdapter(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 20, 16, 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Popular Medicines',
                  style: GoogleFonts.sora(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                GestureDetector(
                  onTap: () {},
                  child: Text(
                    'View all',
                    style: GoogleFonts.sora(
                      fontSize: 13,
                      color: AppColors.primary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final crossAxisCount = constraints.maxWidth < 360 ? 2 : 3;
                final itemHeight = constraints.maxWidth < 360 ? 200.0 : 220.0;

                return GridView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: crossAxisCount,
                    mainAxisExtent: itemHeight,
                    crossAxisSpacing: 12,
                    mainAxisSpacing: 12,
                  ),
                  itemCount: displayMedicines.length,
                  itemBuilder: (context, index) {
                    final medicine = displayMedicines[index];
                    return HomeMedicineCard(
                      medicine: medicine,
                      onTap: () => onMedicineTap(medicine),
                      onAddToCart: () => onAddToCart(medicine),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
