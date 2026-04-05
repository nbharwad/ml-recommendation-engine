import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';
import '../../core/utils/helpers.dart';
import '../../models/medicine_model.dart';

class SearchResultCard extends StatelessWidget {
  final MedicineModel medicine;
  final VoidCallback onTap;
  final VoidCallback onAddToCart;

  const SearchResultCard({
    super.key,
    required this.medicine,
    required this.onTap,
    required this.onAddToCart,
  });

  IconData _getCategoryIcon(String category) {
    return switch (category) {
      'fever' => Icons.thermostat_rounded,
      'pain' => Icons.healing_rounded,
      'skin' => Icons.spa_rounded,
      'diabetes' => Icons.bloodtype_rounded,
      'heart' => Icons.favorite_rounded,
      'vitamins' => Icons.energy_savings_leaf_rounded,
      _ => Icons.medication_rounded,
    };
  }

  @override
  Widget build(BuildContext context) {
    final discount = Helpers.calculateDiscount(medicine.price, medicine.mrp);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFE8ECE7)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.04),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                color: AppColors.primaryLight,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Center(
                child: Icon(
                  _getCategoryIcon(medicine.category),
                  size: 32,
                  color: AppColors.primary,
                ),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          medicine.name,
                          style: GoogleFonts.sora(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            color: AppColors.textPrimary,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      if (medicine.requiresPrescription)
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 3),
                          decoration: BoxDecoration(
                            color: AppColors.error.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(
                            'Rx',
                            style: GoogleFonts.sora(
                              fontSize: 10,
                              color: AppColors.error,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    medicine.manufacturer,
                    style: GoogleFonts.sora(
                      fontSize: 12,
                      color: AppColors.textSecondary,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      Text(
                        '\u20B9${medicine.price.toStringAsFixed(0)}',
                        style: GoogleFonts.sora(
                          fontSize: 15,
                          color: AppColors.primary,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      if (medicine.mrp > medicine.price) ...[
                        Text(
                          '\u20B9${medicine.mrp.toStringAsFixed(0)}',
                          style: GoogleFonts.sora(
                            fontSize: 12,
                            color: AppColors.textSecondary,
                            decoration: TextDecoration.lineThrough,
                          ),
                        ),
                      ],
                      if (discount > 0)
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: AppColors.error.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            '$discount% off',
                            style: GoogleFonts.sora(
                              fontSize: 10,
                              color: AppColors.error,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            ElevatedButton(
              onPressed: onAddToCart,
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 0,
              ),
              child: Text(
                'Add',
                style: GoogleFonts.sora(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class SearchResultsList extends StatelessWidget {
  final List<MedicineModel> medicines;
  final String query;
  final Function(MedicineModel) onMedicineTap;
  final Function(MedicineModel) onAddToCart;
  final VoidCallback onRefresh;

  const SearchResultsList({
    super.key,
    required this.medicines,
    required this.query,
    required this.onMedicineTap,
    required this.onAddToCart,
    required this.onRefresh,
  });

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      color: AppColors.primary,
      backgroundColor: Colors.white,
      strokeWidth: 2.5,
      onRefresh: () async {
        onRefresh();
        await Future.delayed(const Duration(milliseconds: 500));
      },
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: Text(
              '${medicines.length} result${medicines.length == 1 ? '' : 's'} for "$query"',
              style: GoogleFonts.sora(
                fontSize: 13,
                color: AppColors.textSecondary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              physics: const AlwaysScrollableScrollPhysics(),
              itemCount: medicines.length,
              separatorBuilder: (_, __) => const SizedBox(height: 4),
              itemBuilder: (context, index) {
                final medicine = medicines[index];
                return SearchResultCard(
                  medicine: medicine,
                  onTap: () => onMedicineTap(medicine),
                  onAddToCart: () => onAddToCart(medicine),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
