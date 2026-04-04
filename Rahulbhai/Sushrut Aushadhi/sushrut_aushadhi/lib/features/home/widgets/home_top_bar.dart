import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../core/constants/app_colors.dart';
import '../../../providers/notification_provider.dart';
import '../../../providers/auth_provider.dart';
import '../../../core/widgets/notification_panel.dart';

class HomeTopBar extends ConsumerWidget {
  const HomeTopBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      child: Column(
        children: [
          Row(
            children: [
              const Icon(Icons.location_on_rounded, color: AppColors.primary, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Deliver to',
                        style: GoogleFonts.sora(
                            fontSize: 11,
                            color: AppColors.textSecondary,
                            fontWeight: FontWeight.w500)),
                    Consumer(
                      builder: (context, ref, _) {
                        final userAsync = ref.watch(currentUserProvider);
                        return userAsync.when(
                          data: (user) {
                            if (user?.address != null && user!.address.isNotEmpty) {
                              return Text(user.address,
                                  style: GoogleFonts.sora(
                                      fontSize: 15,
                                      color: AppColors.textPrimary,
                                      fontWeight: FontWeight.w700));
                            }
                            return Text('Select delivery location',
                                style: GoogleFonts.sora(
                                    fontSize: 15,
                                    color: AppColors.textPrimary,
                                    fontWeight: FontWeight.w700));
                          },
                          loading: () => Text('Loading...',
                              style: GoogleFonts.sora(
                                  fontSize: 15,
                                  color: AppColors.textPrimary,
                                  fontWeight: FontWeight.w700)),
                          error: (_, __) => Text('Select delivery location',
                              style: GoogleFonts.sora(
                                  fontSize: 15,
                                  color: AppColors.textPrimary,
                                  fontWeight: FontWeight.w700)),
                        );
                      },
                    ),
                    const SizedBox(height: 2),
                  ],
                ),
              ),
              Consumer(
                builder: (context, ref, _) {
                  final unread = ref.watch(unreadCountProvider);
                  return GestureDetector(
                    onTap: () {
                      showDialog(
                        context: context,
                        barrierColor: Colors.black26,
                        builder: (_) => Dialog(
                          alignment: Alignment.topRight,
                          insetPadding: const EdgeInsets.only(top: 60, right: 8),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: const NotificationPanel(),
                        ),
                      );
                    },
                    child: Stack(
                      children: [
                        Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFFF5F7F5),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.notifications_outlined,
                              size: 20, color: Color(0xFF1A1A1A)),
                        ),
                        if (unread > 0)
                          Positioned(
                            top: 2,
                            right: 2,
                            child: Container(
                              width: 14,
                              height: 14,
                              decoration: BoxDecoration(
                                color: const Color(0xFFE53935),
                                shape: BoxShape.circle,
                                border: Border.all(color: Colors.white, width: 1.5),
                              ),
                              child: Center(
                                child: Text(
                                  unread > 9 ? '9+' : '$unread',
                                  style: const TextStyle(
                                    fontSize: 8,
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  );
                },
              )
            ],
          ),
          const SizedBox(height: 14),
          InkWell(
            onTap: () {
              showSearch<void>(
                context: context,
                delegate: _MedicineSearchDelegate(ref: ref),
              );
            },
            borderRadius: BorderRadius.circular(14),
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
                  const Icon(Icons.search_rounded, color: Color(0xFF98A09B)),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text('Search medicines, health products...',
                        style: GoogleFonts.sora(
                            fontSize: 13, color: const Color(0xFF98A09B))),
                  ),
                  Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                        color: AppColors.primary,
                        borderRadius: BorderRadius.circular(12)),
                    child: const Icon(Icons.mic_none_rounded,
                        color: Colors.white, size: 20),
                  )
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MedicineSearchDelegate extends SearchDelegate<void> {
  _MedicineSearchDelegate({required WidgetRef ref})
      : _ref = ref,
        super(searchFieldLabel: 'Search medicines, health products...');

  final WidgetRef _ref;

  @override
  ThemeData appBarTheme(BuildContext context) {
    final theme = Theme.of(context);
    return theme.copyWith(
      scaffoldBackgroundColor: AppColors.background,
      appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          surfaceTintColor: Colors.white,
          elevation: 0),
      textTheme: GoogleFonts.soraTextTheme(theme.textTheme),
    );
  }

  @override
  TextStyle? get searchFieldStyle =>
      GoogleFonts.sora(fontSize: 14, color: AppColors.textPrimary);

  @override
  Widget? buildLeading(BuildContext context) => IconButton(
        icon: const Icon(Icons.arrow_back_rounded, color: AppColors.textPrimary),
        onPressed: () => close(context, null),
      );

  @override
  List<Widget>? buildActions(BuildContext context) => [
        IconButton(
            icon: const Icon(Icons.close_rounded, color: AppColors.textPrimary),
            onPressed: () => query = '')
      ];

  @override
  Widget buildResults(BuildContext context) => const Center(
        child: Text('Search functionality'),
      );

  @override
  Widget buildSuggestions(BuildContext context) => buildResults(context);
}