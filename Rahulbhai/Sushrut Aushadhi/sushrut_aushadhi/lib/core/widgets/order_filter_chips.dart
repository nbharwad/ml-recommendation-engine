import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../models/order_model.dart';
import '../constants/app_colors.dart';

class OrderFilterChips extends StatelessWidget {
  final String? selectedFilter;
  final ValueChanged<String?> onFilterSelected;
  final Map<String, int>? counts;

  const OrderFilterChips({
    super.key,
    this.selectedFilter,
    required this.onFilterSelected,
    this.counts,
  });

  static const List<_FilterOption> _filters = [
    _FilterOption(key: 'all', label: 'All'),
    _FilterOption(key: 'pending', label: 'Pending'),
    _FilterOption(key: 'confirmed', label: 'Confirmed'),
    _FilterOption(key: 'preparing', label: 'Preparing'),
    _FilterOption(key: 'outForDelivery', label: 'Out for Delivery'),
    _FilterOption(key: 'delivered', label: 'Delivered'),
    _FilterOption(key: 'cancelled', label: 'Cancelled'),
  ];

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: _filters.map((filter) {
          final isSelected = selectedFilter == filter.key ||
              (selectedFilter == null && filter.key == 'all');
          final count = counts?[filter.key];

          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: FilterChip(
              label: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(filter.label),
                  if (count != null && count > 0) ...[
                    const SizedBox(width: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? Colors.white.withOpacity(0.3)
                            : AppColors.primary.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        count.toString(),
                        style: GoogleFonts.sora(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: isSelected ? Colors.white : AppColors.primary,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
              selected: isSelected,
              onSelected: (_) => onFilterSelected(
                filter.key == 'all' ? null : filter.key,
              ),
              backgroundColor: Colors.white,
              selectedColor: AppColors.primary,
              labelStyle: GoogleFonts.sora(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: isSelected ? Colors.white : AppColors.textPrimary,
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
                side: BorderSide(
                  color: isSelected ? AppColors.primary : Colors.grey.shade300,
                ),
              ),
              padding: const EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 8,
              ),
              showCheckmark: false,
              elevation: 0,
              pressElevation: 2,
            ),
          );
        }).toList(),
      ),
    );
  }

  static List<String> get filterKeys => _filters.map((f) => f.key).toList();

  static String getFilterLabel(String? key) {
    if (key == null || key == 'all') return 'All';
    return _filters.firstWhere((f) => f.key == key).label;
  }

  static OrderStatus? getStatusFromFilter(String? filter) {
    if (filter == null || filter == 'all') return null;
    return OrderStatus.fromString(filter);
  }
}

class _FilterOption {
  final String key;
  final String label;

  const _FilterOption({
    required this.key,
    required this.label,
  });
}
