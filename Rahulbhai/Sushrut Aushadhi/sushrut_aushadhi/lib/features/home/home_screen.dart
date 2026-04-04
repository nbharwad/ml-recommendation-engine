import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shimmer/shimmer.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../core/constants/app_colors.dart';
import '../../core/utils/helpers.dart';
import '../../core/utils/responsive.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../core/widgets/error_state_widget.dart';
import '../../core/widgets/notification_panel.dart';
import '../../models/article_model.dart';
import '../../models/medicine_model.dart';
import '../../providers/articles_provider.dart';
import '../../providers/cart_provider.dart';
import '../../providers/auth_provider.dart';
import '../../providers/medicines_provider.dart';
import '../../providers/notification_provider.dart';
import '../../services/local_data_service.dart';
import '../../core/constants/app_strings.dart';
import '../../services/remote_config_service.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  final _pageController = PageController(viewportFraction: 0.94);
  Timer? _bannerTimer;
  int _currentBanner = 0;
  bool _isLoading = true;
  List<dynamic> _categories = [];
  List<dynamic> _banners = [];
  List<dynamic> _healthTips = [];

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  @override
  void dispose() {
    _bannerTimer?.cancel();
    _pageController.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    final categories = await LocalDataService.getCategories();
    final banners = await LocalDataService.getBanners();
    final tips = await LocalDataService.getHealthTips();
    if (!mounted) return;
    setState(() {
      _categories = categories;
      _banners = banners;
      _healthTips = tips;
      _isLoading = false;
    });
    _startAutoScroll();
  }

  void _startAutoScroll() {
    _bannerTimer?.cancel();
    if (_banners.length <= 1) return;
    _bannerTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      if (!mounted || !_pageController.hasClients) return;
      final next = (_currentBanner + 1) % _banners.length;
      _pageController.animateToPage(next,
          duration: const Duration(milliseconds: 350), curve: Curves.easeInOut);
    });
  }

  void _filter(String category) {
    ref.read(selectedCategoryProvider.notifier).state = category;
  }

  MedicineModel _model(Map<String, dynamic> m) => MedicineModel(
        id: m['id']?.toString() ?? '',
        name: m['name']?.toString() ?? '',
        genericName: m['genericName']?.toString() ?? '',
        manufacturer: m['brand']?.toString() ?? '',
        description: m['description']?.toString() ?? '',
        price: ((m['price'] ?? 0) as num).toDouble(),
        mrp: ((m['mrp'] ?? 0) as num).toDouble(),
        imageUrl: m['imageUrl']?.toString() ?? '',
        category: m['category']?.toString() ?? 'other',
        requiresPrescription: m['requiresPrescription'] == true,
        stock: (m['stock'] ?? 100) as int,
        unit: m['unit']?.toString() ?? '1 strip',
      );

  void _openMedicine(MedicineModel medicine) {
    context.push('/medicine', extra: medicine.toMap());
  }

  void _addToCartModel(MedicineModel medicine) {
    ref.read(cartProvider.notifier).addItem(medicine);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: AppColors.primary,
        content: Text('${medicine.name} added to cart', style: GoogleFonts.sora()),
      ),
    );
  }

  void _openSearch() {
    showSearch<void>(
      context: context,
      delegate: _MedicineSearchDelegate(ref: ref, onTapMedicine: _openMedicine),
    );
  }

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

  double _heroHeightForWidth(double width) {
    return (width * 0.52).clamp(176.0, 220.0);
  }

  String _extractCity(String address) {
    if (address.isEmpty) return 'Select Location';
    final parts = address.split(',');
    if (parts.length >= 2) return parts[1].trim();
    return parts.first.trim();
  }

  Future<void> _onRefresh() async {
    final category = ref.read(selectedCategoryProvider);
    ref.invalidate(medicinesProvider(category));
    await _loadData();
    await Future.delayed(const Duration(milliseconds: 500));
  }

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).padding.bottom;
    final screenHeight = MediaQuery.of(context).size.height;
    // Responsive: base nav height (64) + bottom padding + extra for taller screens
    final navHeight = 64 + bottomInset + (screenHeight > 800 ? 8 : 4);
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        bottom: false,
        child: RefreshIndicator(
          color: AppColors.primary,
          backgroundColor: Colors.white,
          strokeWidth: 2.5,
          onRefresh: _onRefresh,
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: EdgeInsets.only(bottom: navHeight + 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _topBar(),
                _trustBanner(),
                if (!RemoteConfigService.isCurrentlyOpen) _storeClosedBanner(),
                if (_isLoading)
                  _loading()
                else ...[
                  _hero(),
                  _prescriptionCard(),
                  _categorySection(),
                  _offerStrip(),
                  _categoryFilterChips(),
                  _popularSection(),
                  _drugLicenseCard(),
                  _tipsSection(),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

Widget _trustBanner() {
    final badges = [
      _trustBadge(Icons.verified_rounded, 'Licensed'),
      _trustBadge(Icons.medical_services_rounded, 'Pharmacist'),
      _trustBadge(Icons.local_shipping_rounded, '2Hr Delivery'),
      _trustBadge(Icons.inventory_2_rounded, '19K+ Meds'),
    ];

    return Container(
      color: const Color(0xFFE1F5EE),
      child: Row(
        children: badges
            .map((badge) => Expanded(child: badge))
            .toList(),
      ),
    );
  }

  Widget _storeClosedBanner() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      color: const Color(0xFFFFEBEE),
      child: Row(
        children: [
          const Icon(Icons.access_time, color: Colors.red, size: 14),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Store is currently closed. Opens at ${RemoteConfigService.storeOpenTime}. You can still browse!',
              style: TextStyle(
                fontSize: 11,
                color: Colors.red[700],
              ),
            ),
          ),
        ],
      ),
);
  }

  Widget _trustBadge(IconData icon, String label) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 18, color: const Color(0xFF0F6E56)),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(
            fontSize: 9,
            fontWeight: FontWeight.w600,
            color: Color(0xFF0F6E56),
          ),
        ),
      ],
    );
  }

  Widget _dividerV() {
    return Container(
      width: 0.5,
      height: 28,
      margin: const EdgeInsets.symmetric(horizontal: 12),
      color: const Color(0xFF9FE1CB),
    );
  }

  Widget _drugLicenseCard() {
    return Container(
      margin: const EdgeInsets.fromLTRB(14, 0, 14, 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFEDF2ED)),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final verifiedChip = Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0xFFE1F5EE),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Text(
              'Verified',
              style: TextStyle(
                fontSize: 9,
                fontWeight: FontWeight.w700,
                color: Color(0xFF0F6E56),
              ),
            ),
          );

          final details = Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Licensed Pharmacy',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF1A1A1A),
                ),
              ),
              const SizedBox(height: 2),
Text(
                'Drug License No: ${RemoteConfigService.drugLicenseNo}\nGST: ${RemoteConfigService.gstNumber}',
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.grey.shade600,
                  height: 1.5,
                ),
              ),
            ],
          );

          final iconCard = Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: const Color(0xFFE1F5EE),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Center(
              child: Icon(Icons.local_hospital_rounded, color: AppColors.primary, size: 22),
            ),
          );

          if (constraints.maxWidth < 360) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    iconCard,
                    const SizedBox(width: 12),
                    Expanded(child: details),
                  ],
                ),
                const SizedBox(height: 12),
                verifiedChip,
              ],
            );
          }

          return Row(
            children: [
              iconCard,
              const SizedBox(width: 12),
              Expanded(child: details),
              const SizedBox(width: 12),
              verifiedChip,
            ],
          );
        },
      ),
    );
  }
  Widget _topBar() {
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
                child: Consumer(
                  builder: (context, ref, _) {
                    final userAsync = ref.watch(currentUserProvider);
                    final user = userAsync.valueOrNull;
                    final displayAddress = user?.address ?? '';
                    final city = _extractCity(displayAddress);
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Deliver to',
                            style: GoogleFonts.sora(
                                fontSize: 11,
                                color: AppColors.textSecondary,
                                fontWeight: FontWeight.w500)),
                        Text(city.isNotEmpty ? city : 'Select delivery location',
                            style: GoogleFonts.sora(
                                fontSize: 15,
                                color: AppColors.textPrimary,
                                fontWeight: FontWeight.w700)),
                        const SizedBox(height: 2),
                        Text(displayAddress.isNotEmpty 
                            ? displayAddress 
                            : 'Dispatched from Koramangala, Bengaluru',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.sora(
                                fontSize: 11,
                                color: const Color(0xFF7E867F),
                                fontWeight: FontWeight.w500)),
                      ],
                    );
                  },
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
            onTap: _openSearch,
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

  Widget _hero() {
    if (_banners.isEmpty) {
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
                  itemCount: _banners.length,
                  onPageChanged: (index) => setState(() => _currentBanner = index),
                  itemBuilder: (context, index) {
                    final banner = _banners[index] as Map<String, dynamic>;
                    return Padding(
                      padding: EdgeInsets.only(right: index == _banners.length - 1 ? 0 : 12),
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
            _banners.length,
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
  Widget _prescriptionCard() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 6),
      child: InkWell(
        onTap: () => context.push('/prescription'),
        borderRadius: BorderRadius.circular(20),
        child: CustomPaint(
          painter: _DashedBorderPainter(),
          child: Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFFF4FBF6),
              borderRadius: BorderRadius.circular(20),
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final verticalLayout = constraints.maxWidth < 360;
                final iconCard = Container(
                  width: 54,
                  height: 54,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: const Center(
                    child: Icon(
                      Icons.description_outlined,
                      size: 28,
                      color: AppColors.primary,
                    ),
                  ),
                );
                final content = Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'Have a Prescription?',
                        style: GoogleFonts.sora(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        "Upload it and we'll arrange all medicines",
                        style: GoogleFonts.sora(
                          fontSize: 12,
                          height: 1.35,
                          color: AppColors.textSecondary,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
                        decoration: BoxDecoration(
                          color: AppColors.primary,
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Text(
                          'Upload Now',
                          style: GoogleFonts.sora(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
                if (verticalLayout) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      iconCard,
                      const SizedBox(height: 14),
                      Row(children: [content]),
                    ],
                  );
                }
                return Row(
                  children: [
                    iconCard,
                    const SizedBox(width: 14),
                    content,
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }
  Widget _categorySection() {
    final selectedCategory = ref.watch(selectedCategoryProvider);
    return Column(
      children: [
        _header('Shop by Category',
            action: 'See all', onTap: () => _filter('all')),
        SizedBox(
          height: 124,
          child: ListView.separated(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            scrollDirection: Axis.horizontal,
            itemCount: _categories.length,
            separatorBuilder: (_, __) => const SizedBox(width: 14),
            itemBuilder: (context, index) {
              final c = _categories[index] as Map<String, dynamic>;
              final id = c['id']?.toString() ?? '';
              final selected = selectedCategory == id;
              final color = _categoryColor(id);
              return InkWell(
                onTap: () => _filter(id),
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
                        child: Center(child: _categoryIcon(id, selected)),
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
                            fontWeight:
                                selected ? FontWeight.w700 : FontWeight.w500),
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
  Widget _categoryFilterChips() {
    final selectedCategory = ref.watch(selectedCategoryProvider);
    
    final allCategories = [
      {'id': 'all', 'name': 'All'},
      {'id': 'fever', 'name': 'Fever'},
      {'id': 'pain', 'name': 'Pain'},
      {'id': 'skin', 'name': 'Skin'},
      {'id': 'diabetes', 'name': 'Diabetes'},
      {'id': 'heart', 'name': 'Heart'},
      {'id': 'vitamins', 'name': 'Vitamins'},
    ];

    return SizedBox(
      height: 44,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        itemCount: allCategories.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final category = allCategories[index];
          final isSelected = category['id'] == selectedCategory;
          
          return GestureDetector(
            onTap: () => ref.read(selectedCategoryProvider.notifier).state = category['id'] as String,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: isSelected ? const Color(0xFF0F6E56) : const Color(0xFFE1F5EE),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: isSelected ? const Color(0xFF0F6E56) : const Color(0xFF9FE1CB),
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _getFilterChipIcon(category['id'] as String),
                    size: 16,
                    color: isSelected ? Colors.white : const Color(0xFF0F6E56),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    category['name']?.toString() ?? '',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: isSelected ? Colors.white : const Color(0xFF0F6E56),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  IconData _getFilterChipIcon(String id) {
    return switch (id) {
      'all' => Icons.apps_rounded,
      'fever' => Icons.thermostat_rounded,
      'pain' => Icons.healing_rounded,
      'skin' => Icons.spa_rounded,
      'diabetes' => Icons.bloodtype_rounded,
      'heart' => Icons.favorite_rounded,
      'vitamins' => Icons.vaccines_rounded,
      _ => Icons.medication_rounded,
    };
  }

  Widget _offerStrip() {
    final screenWidth = MediaQuery.of(context).size.width;
    final isCompact = screenWidth < 360;
    
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 18, 16, 6),
      padding: EdgeInsets.symmetric(horizontal: isCompact ? 10 : 14, vertical: isCompact ? 10 : 14),
      decoration: BoxDecoration(
          color: const Color(0xFFFFF3D5),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFF3C158))),
      child: isCompact
          ? Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.local_offer_rounded,
                        size: isCompact ? 20 : 24, color: const Color(0xFFFF8A00)),
                    const SizedBox(width: 8),
                    Expanded(child: _offerText(isCompact: isCompact)),
                  ],
                ),
                const SizedBox(height: 10),
                _offerCode(isCompact: isCompact),
              ],
            )
          : Row(
              children: [
                Icon(Icons.local_offer_rounded,
                    size: 24, color: const Color(0xFFFF8A00)),
                const SizedBox(width: 10),
                Expanded(child: _offerText(isCompact: isCompact)),
                const SizedBox(width: 12),
                _offerCode(isCompact: isCompact),
              ],
            ),
    );
  }
  Widget _popularSection() {
    final selectedCategory = ref.watch(selectedCategoryProvider);
    final medicinesAsync = ref.watch(medicinesProvider(selectedCategory));
    return Column(
      children: [
        _header('Popular Medicines',
            action: 'View all', onTap: () => _filter('all')),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: medicinesAsync.when(
            loading: _buildShimmerGrid,
            error: (error, stackTrace) => ErrorStateWidget(
              message: 'Could not load medicines. Check connection.',
              onRetry: () => ref.invalidate(medicinesProvider(selectedCategory)),
            ),
            data: (medicines) {
              if (medicines.isEmpty) {
                return const EmptyStateWidget(
                  emoji: '????',
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
                            onTap: () => _openMedicine(medicine),
                            onAdd: () => _addToCartModel(medicine),
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
                            'View All ???',
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
  Widget _buildShimmerGrid() {
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
  Widget _tipsSection() {
    return Consumer(
      builder: (context, ref, _) {
        final articlesAsync = ref.watch(articlesProvider);

        return articlesAsync.when(
          loading: () => _buildArticlesShimmer(),
          error: (_, __) => const SizedBox.shrink(),
          data: (articles) => Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 10),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'Health Articles',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF1A1A1A),
                      ),
                    ),
                    const Text(
                      'See more',
                      style: TextStyle(
                        fontSize: 11,
                        color: Color(0xFF0F6E56),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                height: 180,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 14),
                  itemCount: articles.length,
                  itemBuilder: (ctx, i) => _articleCard(context, articles[i]),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildArticlesShimmer() {
    return SizedBox(
      height: 180,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        itemCount: 3,
        itemBuilder: (ctx, i) => Container(
          width: 200,
          margin: const EdgeInsets.only(right: 12),
          decoration: BoxDecoration(
            color: Colors.grey[300],
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
    );
  }

  Widget _articleCard(BuildContext context, ArticleModel article) {
    return GestureDetector(
      onTap: () async {
        final uri = Uri.parse(article.url);
        if (await canLaunchUrl(uri)) {
          await launchUrl(uri, mode: LaunchMode.externalApplication);
        }
      },
      child: Container(
        width: 200,
        margin: const EdgeInsets.only(right: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFEDF2ED)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              height: 90,
              decoration: const BoxDecoration(
                color: Color(0xFFE1F5EE),
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(14),
                  topRight: Radius.circular(14),
                ),
              ),
              child: article.imageUrl.isNotEmpty
                  ? ClipRRect(
                      borderRadius: const BorderRadius.only(
                        topLeft: Radius.circular(14),
                        topRight: Radius.circular(14),
                      ),
                      child: Image.network(
                        article.imageUrl,
                        fit: BoxFit.cover,
                        width: double.infinity,
                        errorBuilder: (_, __, ___) =>
                            const Center(child: Text('Ã°Å¸ÂÂ¥', style: TextStyle(fontSize: 32))),
                      ),
                    )
                  : const Center(
                      child: Text('Ã°Å¸ÂÂ¥', style: TextStyle(fontSize: 32)),
                    ),
            ),
            Padding(
              padding: const EdgeInsets.all(10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    article.title,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1A1A1A),
                      height: 1.3,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Text(
                        article.source,
                        style: const TextStyle(
                          fontSize: 9,
                          color: Color(0xFF0F6E56),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const Spacer(),
                      const Text(
                        'Read Ã¢â€ â€™',
                        style: TextStyle(
                          fontSize: 9,
                          color: Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
  Widget _loading() {
    return Shimmer.fromColors(
      baseColor: const Color(0xFFE8ECE7),
      highlightColor: Colors.white,
      child: Column(
        children: [
          Container(
              height: _heroHeightForWidth(
                  MediaQuery.sizeOf(context).width * 0.94),
              margin: const EdgeInsets.fromLTRB(16, 18, 16, 0),
              decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(22))),
          Container(
              height: 110,
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20))),
          SizedBox(
            height: 88,
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: 6,
              separatorBuilder: (_, __) => const SizedBox(width: 12),
              itemBuilder: (_, __) => Column(
                children: [
                  Container(
                      width: 56,
                      height: 56,
                      decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(18))),
                  const SizedBox(height: 8),
                  Container(
                      width: 56,
                      height: 10,
                      decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(8))),
                ],
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final screenWidth = MediaQuery.of(context).size.width;
                final isCompact = screenWidth < 360;
                final itemHeight = isCompact ? 200.0 : 220.0;
                return GridView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      mainAxisExtent: itemHeight,
                      crossAxisSpacing: 14,
                      mainAxisSpacing: 14),
                  itemCount: 4,
                  itemBuilder: (_, __) => Container(
                      decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(18))),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _offerText({bool isCompact = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('20% off on your first order!',
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.sora(
                color: const Color(0xFFB96A00),
                fontSize: isCompact ? 12 : 14,
                fontWeight: FontWeight.w700)),
        const SizedBox(height: 2),
        Text('Apply the code at checkout for instant savings.',
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style:
                GoogleFonts.sora(color: const Color(0xFFB58228), fontSize: isCompact ? 10 : 11)),
      ],
    );
  }

  Widget _offerCode({bool isCompact = false}) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: isCompact ? 8 : 12, vertical: isCompact ? 6 : 8),
      decoration: BoxDecoration(
          color: const Color(0xFFFF8A00),
          borderRadius: BorderRadius.circular(8)),
      child: Text('SUSHRUT20',
          style: GoogleFonts.sora(
              color: Colors.white,
              fontSize: isCompact ? 10 : 11,
              fontWeight: FontWeight.w800)),
    );
  }

  Widget _categoryIcon(String id, bool selected) {
    return Icon(
      switch (id) {
        'fever' => Icons.thermostat_rounded,
        'pain' => Icons.medication_rounded,
        'skin' => Icons.spa_rounded,
        'diabetes' => Icons.bloodtype_rounded,
        'heart' => Icons.favorite_rounded,
        'vitamins' => Icons.health_and_safety_rounded,
        _ => Icons.local_hospital_rounded,
      },
      size: 28,
      color: selected ? Colors.white : _categoryColor(id),
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
}

class _MedicineCard extends StatelessWidget {
  const _MedicineCard(
      {required this.medicine, required this.onTap, required this.onAdd});
  final Map<String, dynamic> medicine;
  final VoidCallback onTap;
  final VoidCallback onAdd;
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
class _MedicineSearchDelegate extends SearchDelegate<void> {
  _MedicineSearchDelegate({required this.ref, required this.onTapMedicine})
      : super(searchFieldLabel: 'Search medicines, health products...');

  final WidgetRef ref;
  final ValueChanged<MedicineModel> onTapMedicine;

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
  Widget buildResults(BuildContext context) => _SearchBody(
        query: query,
        onTapMedicine: onTapMedicine,
        onAdd: (medicine) {
          ref.read(cartProvider.notifier).addItem(medicine);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
                backgroundColor: AppColors.primary,
                content: Text('${medicine.name} added to cart',
                    style: GoogleFonts.sora())),
          );
        },
      );

  @override
  Widget buildSuggestions(BuildContext context) => buildResults(context);
}

class _SearchBody extends ConsumerWidget {
  const _SearchBody(
      {required this.query,
      required this.onTapMedicine,
      required this.onAdd});

  final String query;
  final ValueChanged<MedicineModel> onTapMedicine;
  final ValueChanged<MedicineModel> onAdd;

  @override
  Widget build(BuildContext context, WidgetRef widgetRef) {
    final trimmedQuery = query.trim();
    if (query.trim().isEmpty) {
      return Center(
          child: Text('Search for medicines',
              style: GoogleFonts.sora(fontSize: 14, color: AppColors.textSecondary)));
    }
    final resultsAsync = widgetRef.watch(searchResultsProvider(trimmedQuery));
    return resultsAsync.when(
      loading: () => const Center(
        child: CircularProgressIndicator(color: AppColors.primary),
      ),
      error: (error, stackTrace) => Center(
        child: Text(
          'Could not search medicines. Check connection.',
          style: GoogleFonts.sora(fontSize: 14, color: AppColors.textSecondary),
        ),
      ),
      data: (items) {
        if (items.isEmpty) {
          return EmptyStateWidget(
            emoji: 'Ã°Å¸â€Â',
            title: 'No Results Found',
            subtitle: 'We couldn\'t find any medicines matching "$query".',
          );
        }
        return ListView.separated(
          padding: const EdgeInsets.all(16),
          itemCount: items.length,
          separatorBuilder: (_, __) => const SizedBox(height: 12),
          itemBuilder: (context, index) {
            final medicine = items[index];
            final discount =
                Helpers.calculateDiscount(medicine.price, medicine.mrp);
            return InkWell(
              onTap: () => onTapMedicine(medicine),
              borderRadius: BorderRadius.circular(16),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: const Color(0xFFE8ECE7))),
                child: Row(
                  children: [
                    Container(
                      width: 66,
                      height: 66,
                      decoration: BoxDecoration(
                          color: const Color(0xFFF3F5F2),
                          borderRadius: BorderRadius.circular(16)),
                      child: Center(
                        child: Icon(
                          _searchResultIcon(medicine.category),
                          size: 32,
                          color: AppColors.primary,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Expanded(
                                  child: Text(medicine.name,
                                      style: GoogleFonts.sora(
                                          fontSize: 14,
                                          color: AppColors.textPrimary,
                                          fontWeight: FontWeight.w700))),
                              if (medicine.requiresPrescription)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 6, vertical: 3),
                                  decoration: BoxDecoration(
                                      color: const Color(0xFFFFEBEE),
                                      borderRadius: BorderRadius.circular(8)),
                                  child: Text('Rx',
                                      style: GoogleFonts.sora(
                                          fontSize: 10,
                                          color: AppColors.discountRed,
                                          fontWeight: FontWeight.w800)),
                                ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(medicine.manufacturer,
                              style: GoogleFonts.sora(
                                  fontSize: 11, color: AppColors.textSecondary)),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 6,
                            children: [
                              Text('\u20B9${medicine.price.toStringAsFixed(0)}',
                                  style: GoogleFonts.sora(
                                      fontSize: 14,
                                      color: AppColors.primary,
                                      fontWeight: FontWeight.w800)),
                              Text('\u20B9${medicine.mrp.toStringAsFixed(0)}',
                                  style: GoogleFonts.sora(
                                      fontSize: 11,
                                      color: AppColors.textSecondary,
                                      decoration: TextDecoration.lineThrough)),
                              if (discount > 0)
                                Text('$discount% off',
                                    style: GoogleFonts.sora(
                                        fontSize: 11,
                                        color: AppColors.discountRed,
                                        fontWeight: FontWeight.w700)),
                            ],
                          ),
                        ],
                      ),
                    ),
                    ElevatedButton(
                      onPressed: () => onAdd(medicine),
                      style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 10),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12))),
                      child: Text('Add',
                          style: GoogleFonts.sora(
                              fontSize: 12, fontWeight: FontWeight.w700)),
                    )
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  IconData _searchResultIcon(String category) {
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
}
class _HomeBottomNav extends StatelessWidget {
  const _HomeBottomNav(
      {required this.currentIndex,
      required this.cartCount,
      required this.onTap});

  final int currentIndex;
  final int cartCount;
  final ValueChanged<int> onTap;

  @override
  Widget build(BuildContext context) {
    return BottomNavigationBar(
      currentIndex: currentIndex,
      onTap: onTap,
      type: BottomNavigationBarType.fixed,
      backgroundColor: Colors.white,
      selectedItemColor: AppColors.primary,
      unselectedItemColor: const Color(0xFF9BA29D),
      selectedLabelStyle:
          GoogleFonts.sora(fontSize: 11, fontWeight: FontWeight.w700),
      unselectedLabelStyle:
          GoogleFonts.sora(fontSize: 11, fontWeight: FontWeight.w500),
      items: [
        _item(0, Icons.home_rounded, 'Home'),
        _item(1, Icons.search_rounded, 'Search'),
        _item(2, Icons.shopping_bag_outlined, 'Cart', badge: cartCount),
        _item(3, Icons.receipt_long_outlined, 'Orders'),
        _item(4, Icons.person_outline_rounded, 'Profile'),
      ],
    );
  }

  BottomNavigationBarItem _item(int index, IconData icon, String label,
      {int badge = 0}) {
    final active = index == currentIndex;
    return BottomNavigationBarItem(
      label: label,
      icon: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Stack(
            clipBehavior: Clip.none,
            children: [
              Icon(icon, color: active ? AppColors.primary : const Color(0xFF9BA29D)),
              if (badge > 0)
                Positioned(
                  top: -6,
                  right: -10,
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                    decoration: BoxDecoration(
                        color: AppColors.discountRed,
                        borderRadius: BorderRadius.circular(999)),
                    child: Text('$badge',
                        style: GoogleFonts.sora(
                            color: Colors.white,
                            fontSize: 9,
                            fontWeight: FontWeight.w700)),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 4),
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
                color: active ? AppColors.primary : Colors.transparent,
                shape: BoxShape.circle),
          )
        ],
      ),
    );
  }
}

class _DashedBorderPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final path = Path()
      ..addRRect(RRect.fromRectAndRadius(
          Offset.zero & size, const Radius.circular(20)));
    final paint = Paint()
      ..color = const Color(0xFFA9D9C6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    for (final metric in path.computeMetrics()) {
      double distance = 0;
      while (distance < metric.length) {
        canvas.drawPath(metric.extractPath(distance, distance + 7), paint);
        distance += 12;
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}


















