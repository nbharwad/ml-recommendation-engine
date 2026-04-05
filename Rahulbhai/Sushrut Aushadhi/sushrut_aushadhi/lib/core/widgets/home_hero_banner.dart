import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../constants/app_colors.dart';

class HomeHeroBanner extends StatefulWidget {
  final List<Map<String, dynamic>> banners;
  final PageController pageController;
  final int currentBanner;
  final ValueChanged<int> onPageChanged;

  const HomeHeroBanner({
    super.key,
    required this.banners,
    required this.pageController,
    required this.currentBanner,
    required this.onPageChanged,
  });

  @override
  State<HomeHeroBanner> createState() => _HomeHeroBannerState();
}

class _HomeHeroBannerState extends State<HomeHeroBanner> {
  @override
  Widget build(BuildContext context) {
    if (widget.banners.isEmpty) {
      return const SliverToBoxAdapter(child: SizedBox.shrink());
    }

    return SliverToBoxAdapter(
      child: Column(
        children: [
          const SizedBox(height: 16),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final bannerAspectRatio = constraints.maxWidth < 380
                    ? 1.08
                    : constraints.maxWidth < 600
                        ? 1.55
                        : 2.25;
                return AspectRatio(
                  aspectRatio: bannerAspectRatio,
                  child: PageView.builder(
                    controller: widget.pageController,
                    itemCount: widget.banners.length,
                    onPageChanged: widget.onPageChanged,
                    itemBuilder: (context, index) {
                      final banner =
                          widget.banners[index] as Map<String, dynamic>;
                      return Padding(
                        padding: EdgeInsets.only(
                            right: index == widget.banners.length - 1 ? 0 : 12),
                        child: _BannerCard(
                          banner: banner,
                          isEven: index.isEven,
                        ),
                      );
                    },
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 12),
          _buildPageIndicators(),
        ],
      ),
    );
  }

  Widget _buildPageIndicators() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(
        widget.banners.length,
        (i) => AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          margin: const EdgeInsets.symmetric(horizontal: 3),
          width: widget.currentBanner == i ? 18 : 6,
          height: 6,
          decoration: BoxDecoration(
            color: widget.currentBanner == i
                ? AppColors.primary
                : const Color(0xFFD8DDD8),
            borderRadius: BorderRadius.circular(99),
          ),
        ),
      ),
    );
  }
}

class _BannerCard extends StatelessWidget {
  final Map<String, dynamic> banner;
  final bool isEven;

  const _BannerCard({
    required this.banner,
    required this.isEven,
  });

  @override
  Widget build(BuildContext context) {
    final imageUrl = banner['imageUrl']?.toString() ?? '';
    final title = banner['title']?.toString() ?? '';
    final subtitle = banner['subtitle']?.toString() ?? '';
    final buttonText = banner['buttonText']?.toString() ?? 'Shop Now';

    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          colors: isEven
              ? [const Color(0xFF0F6E56), const Color(0xFF1D9E75)]
              : [const Color(0xFF0277BD), const Color(0xFF0288D1)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            top: -20,
            right: -20,
            child: Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.1),
              ),
            ),
          ),
          Positioned(
            bottom: -15,
            left: -15,
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.08),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      if (title.isNotEmpty) ...[
                        Text(
                          title,
                          style: GoogleFonts.sora(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 4),
                      ],
                      if (subtitle.isNotEmpty) ...[
                        Text(
                          subtitle,
                          style: GoogleFonts.sora(
                            color: Colors.white.withValues(alpha: 0.85),
                            fontSize: 12,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 12),
                      ],
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          buttonText,
                          style: GoogleFonts.sora(
                            color: isEven
                                ? AppColors.primary
                                : AppColors.labPrimary,
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                if (imageUrl.isNotEmpty)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: CachedNetworkImage(
                      imageUrl: imageUrl,
                      width: 80,
                      height: 80,
                      fit: BoxFit.cover,
                      placeholder: (_, __) => Container(
                        width: 80,
                        height: 80,
                        color: Colors.white.withValues(alpha: 0.2),
                        child: const Icon(
                          Icons.image,
                          color: Colors.white,
                          size: 32,
                        ),
                      ),
                      errorWidget: (_, __, ___) => Container(
                        width: 80,
                        height: 80,
                        color: Colors.white.withValues(alpha: 0.2),
                        child: const Icon(
                          Icons.medication_rounded,
                          color: Colors.white,
                          size: 32,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
