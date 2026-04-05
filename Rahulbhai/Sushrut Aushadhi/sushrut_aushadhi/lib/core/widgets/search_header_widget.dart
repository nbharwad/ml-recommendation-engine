import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../constants/app_colors.dart';

class SearchHeaderWidget extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final ValueChanged<String> onChanged;
  final ValueChanged<String> onSubmitted;
  final VoidCallback onClear;
  final VoidCallback? onVoiceTap;
  final VoidCallback onBack;

  const SearchHeaderWidget({
    super.key,
    required this.controller,
    required this.focusNode,
    required this.onChanged,
    required this.onSubmitted,
    required this.onClear,
    this.onVoiceTap,
    required this.onBack,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      child: SafeArea(
        bottom: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_rounded,
                        color: AppColors.textPrimary),
                    onPressed: onBack,
                  ),
                  Expanded(
                    child: Container(
                      height: 48,
                      decoration: BoxDecoration(
                        color: const Color(0xFFF4F5F3),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0xFFEDEFEA)),
                      ),
                      child: TextField(
                        controller: controller,
                        focusNode: focusNode,
                        onChanged: onChanged,
                        onSubmitted: onSubmitted,
                        style: GoogleFonts.sora(
                            fontSize: 14, color: AppColors.textPrimary),
                        decoration: InputDecoration(
                          hintText: 'Search medicines, health products...',
                          hintStyle: GoogleFonts.sora(
                              fontSize: 14, color: const Color(0xFF98A09B)),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 14),
                          prefixIcon: const Icon(Icons.search_rounded,
                              color: Color(0xFF98A09B), size: 22),
                          suffixIcon: controller.text.isNotEmpty
                              ? IconButton(
                                  icon: const Icon(Icons.close_rounded,
                                      color: AppColors.textSecondary, size: 20),
                                  onPressed: onClear,
                                )
                              : null,
                        ),
                      ),
                    ),
                  ),
                  if (onVoiceTap != null)
                    IconButton(
                      icon: const Icon(Icons.mic_rounded,
                          color: AppColors.primary),
                      onPressed: onVoiceTap,
                    )
                  else
                    const SizedBox(width: 8),
                ],
              ),
            ),
            Container(height: 1, color: Colors.grey.shade200),
          ],
        ),
      ),
    );
  }
}
