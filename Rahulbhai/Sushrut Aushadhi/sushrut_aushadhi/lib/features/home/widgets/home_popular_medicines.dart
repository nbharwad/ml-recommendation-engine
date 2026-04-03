import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shimmer/shimmer.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/utils/helpers.dart';
import '../../../core/widgets/empty_state_widget.dart';
import '../../../core/widgets/error_state_widget.dart';
import '../../../core/utils/responsive.dart';
import '../../../providers/medicines_provider.dart';
import '../../../providers/cart_provider.dart';
import '../../../models/medicine_model.dart';

class HomePopularMedicines extends ConsumerWidget {
  const HomePopularMedicines({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedCategory = ref.watch(selectedCategoryProvider);
    final medicinesAsync = ref.watch(medicinesProvider(selectedCategory));

    return Column(
      children: [
        _header('Popular Medicines',
            action: 'View all', onTap: () => ref.read(selectedCategoryProvider.notifier).state = 'all'),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: medicinesAsync.when(
            loading: () => HomePopularMedicines._buildShimmerGrid(context),
            error: (error, stackTrace) => ErrorStateWidget(
              message: 'Could not load medicines. Check connection.',
              onRetry: () => ref.invalidate(medicinesProvider(selectedCategory)),
            ),
            data: (medicines) {
              if (medicines.isEmpty) {
                return const EmptyStateWidget(
                  emoji: '💊',
                  title: 'No Medicines Found',
                  subtitle: 'We couldn\'t find any medicines in this category.',
                );
              }
              final popular = medicines.take(6).toList();
              return LayoutBuilder(
                builder: (context, constraints) {
                  final crossAxisCount = context.gridCrossAxisCount;
                  final childAspectRatio = context.gridAspectRatio;
                  return Column(
                    children: [
                      GridView.builder(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: crossAxisCount,
                          childAspectRatio: childAspectRatio,
                          crossAxisSpacing: 14,
                          mainAxisSpacing: 14,
                        ),
                        itemCount: popular.length,
                        itemBuilder: (context, index) {
                          final medicine = popular[index];
                          return _MedicineCard(
                            medicine: medicine.toMap(),
                            onTap: () => context.push('/medicine', extra: medicine.toMap()),
                            onAdd: () {
                              ref.read(cartProvider.notifier).addItem(medicine);
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  backgroundColor: AppColors.primary,
                                  content: Text('${medicine.name} added to cart', style: GoogleFonts.sora()),
                                ),
                              );
                            },
                          );
                        },
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        child: TextButton(
                          onPressed: () => context.push('/search'),
                          style: TextButton.styleFrom(
                            foregroundColor: AppColors.primary,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: Text(
                            'View All 💊',
                            style: GoogleFonts.sora(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _header(String title, {String? action, VoidCallback? onTap}) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 12),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final stacked = constraints.maxWidth < 340;
          if (stacked) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.sora(
                    fontSize: 21,
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (action != null) ...[
                  const SizedBox(height: 6),
                  InkWell(
                    onTap: onTap,
                    child: Text(
                      action,
                      style: GoogleFonts.sora(
                        fontSize: 13,
                        color: AppColors.primary,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ],
            );
          }
          return Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: GoogleFonts.sora(
                    fontSize: 21,
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              if (action != null)
                InkWell(
                  onTap: onTap,
                  child: Text(
                    action,
                    style: GoogleFonts.sora(
                      fontSize: 13,
                      color: AppColors.primary,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }

  static Widget _buildShimmerGrid(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final crossAxisCount = context.gridCrossAxisCount;
        return GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            childAspectRatio: context.gridAspectRatio,
            crossAxisSpacing: 14,
            mainAxisSpacing: 14,
          ),
          itemCount: crossAxisCount * 2,
          itemBuilder: (context, index) => Shimmer.fromColors(
            baseColor: Colors.grey[300]!,
            highlightColor: Colors.grey[100]!,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _MedicineCard extends StatelessWidget {
  const _MedicineCard(
      {required this.medicine, required this.onTap, required this.onAdd});

  final Map<String, dynamic> medicine;
  final VoidCallback onTap;
  final VoidCallback onAdd;

  IconData _medicineIcon(String category) {
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
    final price = ((medicine['price'] ?? 0) as num).toDouble();
    final mrp = ((medicine['mrp'] ?? 0) as num).toDouble();
    final discount = Helpers.calculateDiscount(price, mrp);
    return LayoutBuilder(
      builder: (context, constraints) {
        final isCompact = constraints.maxWidth < 170;
        final imageSize = constraints.maxWidth * (isCompact ? 0.42 : 0.36);
        final spacing = isCompact ? 8.0 : 10.0;
        return InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(18),
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
                    if (medicine['requiresPrescription'] == true)
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
                            color: AppColors.discountRed,
                            fontSize: isCompact ? 8 : 9,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      )
                    else
                      const SizedBox.shrink(),
                    const Spacer(),
                    Container(
                      width: isCompact ? 26 : 28,
                      height: isCompact ? 26 : 28,
                      decoration: BoxDecoration(
                        color: const Color(0xFFF8F8F6),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Icon(
                        Icons.favorite_border_rounded,
                        size: isCompact ? 14 : 15,
                        color: const Color(0xFF8D948F),
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
                        _medicineIcon(medicine['category']?.toString() ?? ''),
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
                      Wrap(
                        spacing: 4,
                        runSpacing: 2,
                        children: [
                          FittedBox(
                            fit: BoxFit.scaleDown,
                            child: Text(
                              '\u20B9${price.toStringAsFixed(0)}',
                              style: GoogleFonts.sora(
                                color: AppColors.primary,
                                fontSize: isCompact ? 13 : 14,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ),
                          FittedBox(
                            fit: BoxFit.scaleDown,
                            child: Text(
                              '\u20B9${mrp.toStringAsFixed(0)}',
                              style: GoogleFonts.sora(
                                color: AppColors.textSecondary,
                                fontSize: isCompact ? 9 : 10,
                                decoration: TextDecoration.lineThrough,
                              ),
                            ),
                          ),
                          if (discount > 0)
                            FittedBox(
                              fit: BoxFit.scaleDown,
                              child: Text(
                                '$discount% off',
                                style: GoogleFonts.sora(
                                  color: AppColors.discountRed,
                                  fontSize: isCompact ? 9 : 10,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
                SizedBox(height: spacing),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: onAdd,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: EdgeInsets.symmetric(vertical: isCompact ? 8 : 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: Text(
                      'Add to Cart',
                      style: GoogleFonts.sora(
                        fontSize: isCompact ? 11 : 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
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