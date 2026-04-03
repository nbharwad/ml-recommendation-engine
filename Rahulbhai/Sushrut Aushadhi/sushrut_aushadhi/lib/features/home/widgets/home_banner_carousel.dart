import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../core/constants/app_colors.dart';

class HomeBannerCarousel extends StatefulWidget {
  const HomeBannerCarousel({super.key, required this.banners});

  final List<dynamic> banners;

  @override
  State<HomeBannerCarousel> createState() => _HomeBannerCarouselState();
}

class _HomeBannerCarouselState extends State<HomeBannerCarousel> {
  final _pageController = PageController(viewportFraction: 0.94);
  Timer? _bannerTimer;
  int _currentBanner = 0;

  @override
  void initState() {
    super.initState();
    _startAutoScroll();
  }

  @override
  void dispose() {
    _bannerTimer?.cancel();
    _pageController.dispose();
    super.dispose();
  }

  void _startAutoScroll() {
    _bannerTimer?.cancel();
    if (widget.banners.length <= 1) return;
    _bannerTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      if (!mounted || !_pageController.hasClients) return;
      final next = (_currentBanner + 1) % widget.banners.length;
      _pageController.animateToPage(next,
          duration: const Duration(milliseconds: 350), curve: Curves.easeInOut);
    });
  }

  @override
  Widget build(BuildContext context) {
    if (widget.banners.isEmpty) {
      return const SizedBox.shrink();
    }
    return Column(
      children: [
        const SizedBox(height: 18),
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
                  controller: _pageController,
                  itemCount: widget.banners.length,
                  onPageChanged: (index) => setState(() => _currentBanner = index),
                  itemBuilder: (context, index) {
                    final banner = widget.banners[index] as Map<String, dynamic>;
                    return Padding(
                      padding: EdgeInsets.only(right: index == widget.banners.length - 1 ? 0 : 12),
                      child: _ResponsiveBannerCard(
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
        const SizedBox(height: 10),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            widget.banners.length,
            (i) => AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              margin: const EdgeInsets.symmetric(horizontal: 3),
              width: _currentBanner == i ? 18 : 6,
              height: 6,
              decoration: BoxDecoration(
                color: _currentBanner == i ? AppColors.primary : const Color(0xFFD8DDD8),
                borderRadius: BorderRadius.circular(99),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _ResponsiveBannerCard extends StatelessWidget {
  const _ResponsiveBannerCard({required this.banner, required this.isEven});

  final Map<String, dynamic> banner;
  final bool isEven;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final compact = constraints.maxWidth < 320;
        final contentWidth = constraints.maxWidth * (compact ? 0.6 : 0.52);
        return Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(22),
            gradient: const LinearGradient(
              colors: [Color(0xFF0F6E56), Color(0xFF4ECBA5)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: Stack(
            children: [
              Positioned(
                right: -24,
                top: -10,
                child: Container(
                  width: compact ? 96 : 128,
                  height: compact ? 96 : 128,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.08),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
              Positioned(
                right: compact ? 14 : 20,
                bottom: compact ? 14 : 18,
                child: Transform.rotate(
                  angle: 0.2,
                  child: Icon(
                    isEven
                        ? Icons.local_shipping_rounded
                        : Icons.health_and_safety_rounded,
                    size: compact ? 40 : 54,
                    color: Colors.white.withOpacity(0.92),
                  ),
                ),
              ),
              Padding(
                padding: EdgeInsets.all(compact ? 14 : 18),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: EdgeInsets.symmetric(
                        horizontal: compact ? 8 : 10,
                        vertical: compact ? 4 : 5,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        (banner['tag'] ?? '').toString().toUpperCase(),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.sora(
                          color: Colors.white,
                          fontSize: compact ? 9 : 10,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    const Spacer(),
                    SizedBox(
                      width: contentWidth,
                      child: Text(
                        banner['title']?.toString() ?? '',
                        maxLines: compact ? 2 : 3,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.sora(
                          color: Colors.white,
                          fontSize: compact ? 18 : 22,
                          height: 1.12,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    SizedBox(height: compact ? 4 : 6),
                    SizedBox(
                      width: contentWidth,
                      child: Text(
                        banner['subtitle']?.toString() ?? '',
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.sora(
                          color: Colors.white.withOpacity(0.76),
                          fontSize: compact ? 11 : 12,
                          height: 1.3,
                        ),
                      ),
                    ),
                    SizedBox(height: compact ? 8 : 12),
                    Container(
                      padding: EdgeInsets.symmetric(
                        horizontal: compact ? 13 : 16,
                        vertical: compact ? 8 : 9,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        "${banner['btnText'] ?? 'Order Now'} ->",
                        style: GoogleFonts.sora(
                          color: AppColors.primary,
                          fontSize: compact ? 11 : 12,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}