import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

class HomeSectionSwitcher extends StatelessWidget {
  final int activeSection;
  final ValueChanged<int> onSectionChanged;

  const HomeSectionSwitcher({
    super.key,
    required this.activeSection,
    required this.onSectionChanged,
  });

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: Container(
        color: Colors.white,
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
        child: Container(
          height: 48,
          decoration: BoxDecoration(
            color: const Color(0xFFF4F5F3),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Row(
            children: [
              Expanded(
                child: _SwitcherTab(
                  icon: Icons.medication_rounded,
                  label: 'Medicines',
                  isActive: activeSection == 0,
                  activeColor: AppColors.primary,
                  onTap: () => onSectionChanged(0),
                ),
              ),
              Expanded(
                child: _SwitcherTab(
                  icon: Icons.biotech_rounded,
                  label: 'Lab Tests',
                  isActive: activeSection == 1,
                  activeColor: AppColors.labPrimary,
                  onTap: () => onSectionChanged(1),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SwitcherTab extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isActive;
  final Color activeColor;
  final VoidCallback onTap;

  const _SwitcherTab({
    required this.icon,
    required this.label,
    required this.isActive,
    required this.activeColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          color: isActive ? activeColor : Colors.transparent,
          borderRadius: BorderRadius.circular(11),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 18,
              color: isActive ? Colors.white : AppColors.textSecondary,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: GoogleFonts.sora(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: isActive ? Colors.white : AppColors.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
