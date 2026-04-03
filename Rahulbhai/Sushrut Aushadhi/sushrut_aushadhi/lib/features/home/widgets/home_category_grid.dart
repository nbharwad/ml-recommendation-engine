import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../core/constants/app_colors.dart';
import '../../../providers/medicines_provider.dart';

class HomeCategoryGrid extends ConsumerWidget {
  const HomeCategoryGrid({super.key, required this.categories});

  final List<dynamic> categories;

  Color _categoryColor(String id) {
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

  IconData _categoryIcon(String id) {
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
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedCategory = ref.watch(selectedCategoryProvider);

    return Column(
      children: [
        _header('Shop by Category',
            action: 'See all', onTap: () => ref.read(selectedCategoryProvider.notifier).state = 'all'),
        SizedBox(
          height: 124,
          child: ListView.separated(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            scrollDirection: Axis.horizontal,
            itemCount: categories.length,
            separatorBuilder: (_, __) => const SizedBox(width: 14),
            itemBuilder: (context, index) {
              final c = categories[index] as Map<String, dynamic>;
              final id = c['id']?.toString() ?? '';
              final selected = selectedCategory == id;
              final color = _categoryColor(id);
              return InkWell(
                onTap: () => ref.read(selectedCategoryProvider.notifier).state = id,
                borderRadius: BorderRadius.circular(18),
                child: SizedBox(
                  width: 74,
                  child: Column(
                    children: [
                      Container(
                        width: 56,
                        height: 56,
                        decoration: BoxDecoration(
                            color: selected ? color : color.withOpacity(0.18),
                            borderRadius: BorderRadius.circular(18)),
                        child: Center(
                          child: Icon(
                            _categoryIcon(id),
                            size: 28,
                            color: selected ? Colors.white : color,
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        c['name']?.toString() ?? '',
                        maxLines: 2,
                        textAlign: TextAlign.center,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.sora(
                            fontSize: 11,
                            height: 1.2,
                            color: selected ? AppColors.textPrimary : AppColors.textSecondary,
                            fontWeight: selected ? FontWeight.w700 : FontWeight.w500),
                      )
                    ],
                  ),
                ),
              );
            },
          ),
        )
      ],
    );
  }

  Widget _header(String title, {String? action, VoidCallback? onTap}) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 12),
      child: Row(
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
      ),
    );
  }
}