import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

class HomeCategoryChipRow extends StatelessWidget {
  final String selectedCategory;
  final ValueChanged<String> onCategorySelected;

  const HomeCategoryChipRow({
    super.key,
    required this.selectedCategory,
    required this.onCategorySelected,
  });

  static const List<Map<String, String>> categories = [
    {'id': 'all', 'name': 'All', 'icon': 'apps'},
    {'id': 'fever', 'name': 'Fever', 'icon': 'thermostat'},
    {'id': 'pain', 'name': 'Pain', 'icon': 'healing'},
    {'id': 'skin', 'name': 'Skin', 'icon': 'spa'},
    {'id': 'diabetes', 'name': 'Diabetes', 'icon': 'bloodtype'},
    {'id': 'heart', 'name': 'Heart', 'icon': 'favorite'},
    {'id': 'vitamins', 'name': 'Vitamins', 'icon': 'vaccines'},
  ];

  IconData _getIcon(String iconName) {
    switch (iconName) {
      case 'apps':
        return Icons.apps_rounded;
      case 'thermostat':
        return Icons.thermostat_rounded;
      case 'healing':
        return Icons.healing_rounded;
      case 'spa':
        return Icons.spa_rounded;
      case 'bloodtype':
        return Icons.bloodtype_rounded;
      case 'favorite':
        return Icons.favorite_rounded;
      case 'vaccines':
        return Icons.vaccines_rounded;
      default:
        return Icons.medication_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: SizedBox(
        height: 48,
        child: ListView.separated(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          scrollDirection: Axis.horizontal,
          itemCount: categories.length,
          separatorBuilder: (_, __) => const SizedBox(width: 8),
          itemBuilder: (context, index) {
            final category = categories[index];
            final isSelected = category['id'] == selectedCategory;

            return GestureDetector(
              onTap: () => onCategorySelected(category['id']!),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding:
                    const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color:
                      isSelected ? AppColors.primary : AppColors.primaryLight,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: isSelected
                        ? AppColors.primary
                        : const Color(0xFF9FE1CB),
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _getIcon(category['icon']!),
                      size: 16,
                      color: isSelected ? Colors.white : AppColors.primary,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      category['name']!,
                      style: GoogleFonts.sora(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: isSelected ? Colors.white : AppColors.primary,
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class HomeCategoriesSection extends StatelessWidget {
  final List<Map<String, dynamic>> categories;
  final String selectedCategory;
  final ValueChanged<String> onCategoryTap;

  const HomeCategoriesSection({
    super.key,
    required this.categories,
    required this.selectedCategory,
    required this.onCategoryTap,
  });

  Color _getCategoryColor(String id) {
    switch (id) {
      case 'fever':
        return const Color(0xFFF7A34B);
      case 'pain':
        return const Color(0xFF4ABF84);
      case 'skin':
        return const Color(0xFF76B8FF);
      case 'diabetes':
        return const Color(0xFFF39BB8);
      case 'heart':
        return const Color(0xFFA284FF);
      case 'vitamins':
        return const Color(0xFF42C7B2);
      default:
        return AppColors.primary;
    }
  }

  IconData _getCategoryIcon(String id) {
    switch (id) {
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
        return Icons.local_hospital_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
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
                  'Shop by Category',
                  style: GoogleFonts.sora(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                GestureDetector(
                  onTap: () => onCategoryTap('all'),
                  child: Text(
                    'See all',
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
          SizedBox(
            height: 110,
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: categories.length,
              separatorBuilder: (_, __) => const SizedBox(width: 14),
              itemBuilder: (context, index) {
                final c = categories[index] as Map<String, dynamic>;
                final id = c['id']?.toString() ?? '';
                final selected = selectedCategory == id;
                final color = _getCategoryColor(id);

                return GestureDetector(
                  onTap: () => onCategoryTap(id),
                  child: SizedBox(
                    width: 74,
                    child: Column(
                      children: [
                        Container(
                          width: 56,
                          height: 56,
                          decoration: BoxDecoration(
                            color: selected
                                ? color
                                : color.withValues(alpha: 0.18),
                            borderRadius: BorderRadius.circular(18),
                          ),
                          child: Center(
                            child: Icon(
                              _getCategoryIcon(id),
                              size: 28,
                              color: selected ? Colors.white : color,
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          c['name']?.toString() ?? '',
                          maxLines: 2,
                          textAlign: TextAlign.center,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.sora(
                            fontSize: 11,
                            height: 1.2,
                            color: selected
                                ? AppColors.textPrimary
                                : AppColors.textSecondary,
                            fontWeight:
                                selected ? FontWeight.w700 : FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
