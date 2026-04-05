import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shimmer/shimmer.dart';

import '../../core/constants/app_colors.dart';
import '../../core/widgets/home_header_widget.dart';
import '../../core/widgets/home_trust_banner.dart';
import '../../core/widgets/home_section_switcher.dart';
import '../../core/widgets/home_hero_banner.dart';
import '../../core/widgets/home_prescription_card.dart';
import '../../core/widgets/home_category_chip_row.dart';
import '../../core/widgets/home_medicine_card.dart';
import '../../core/widgets/home_article_card.dart';
import '../../core/widgets/home_offer_strip.dart';
import '../../core/widgets/home_drug_license_card.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../core/widgets/error_state_widget.dart';
import '../../models/medicine_model.dart';
import '../../providers/articles_provider.dart';
import '../../providers/cart_provider.dart';
import '../../providers/medicines_provider.dart';
import '../../services/local_data_service.dart';
import '../../services/remote_config_service.dart';
import '../lab/widgets/lab_home_widget.dart';

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
  int _activeSection = 0;
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
        content:
            Text('${medicine.name} added to cart', style: GoogleFonts.sora()),
      ),
    );
  }

  void _openSearch() {
    showSearch<void>(
      context: context,
      delegate: _MedicineSearchDelegate(ref: ref, onTapMedicine: _openMedicine),
    );
  }

  Future<void> _onRefresh() async {
    final category = ref.read(selectedCategoryProvider);
    ref.invalidate(medicinesProvider(category));
    await _loadData();
    await Future.delayed(const Duration(milliseconds: 500));
  }

  void _onSectionChanged(int section) {
    setState(() => _activeSection = section);
  }

  void _onCategorySelected(String category) {
    _filter(category);
  }

  @override
  Widget build(BuildContext context) {
    final selectedCategory = ref.watch(selectedCategoryProvider);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: RefreshIndicator(
        color: AppColors.primary,
        backgroundColor: Colors.white,
        strokeWidth: 2.5,
        onRefresh: _onRefresh,
        child: CustomScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          slivers: [
            HomeHeaderWidget(
              onSearchTap: _openSearch,
              onLocationTap: () {},
            ),
            const HomeTrustBanner(),
            HomeSectionSwitcher(
              activeSection: _activeSection,
              onSectionChanged: _onSectionChanged,
            ),
            if (!RemoteConfigService.isCurrentlyOpen)
              StoreClosedBanner(
                  storeOpenTime: RemoteConfigService.storeOpenTime),
            if (_activeSection == 1)
              const SliverToBoxAdapter(child: LabHomeWidget())
            else if (_isLoading)
              _buildLoadingState()
            else ...[
              HomeHeroBanner(
                banners:
                    _banners.map((e) => e as Map<String, dynamic>).toList(),
                pageController: _pageController,
                currentBanner: _currentBanner,
                onPageChanged: (index) =>
                    setState(() => _currentBanner = index),
              ),
              HomePrescriptionCard(
                onTap: () => context.push('/prescription?type=medicine'),
              ),
              HomeCategoriesSection(
                categories:
                    _categories.map((e) => e as Map<String, dynamic>).toList(),
                selectedCategory: selectedCategory,
                onCategoryTap: _filter,
              ),
              const HomeOfferStrip(),
              HomeCategoryChipRow(
                selectedCategory: selectedCategory,
                onCategorySelected: _onCategorySelected,
              ),
              _buildPopularMedicines(selectedCategory),
              const HomeDrugLicenseCard(),
              _buildArticlesSection(),
            ],
            SliverToBoxAdapter(
              child:
                  SizedBox(height: MediaQuery.of(context).padding.bottom + 80),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingState() {
    return SliverToBoxAdapter(
      child: Shimmer.fromColors(
        baseColor: const Color(0xFFE8ECE7),
        highlightColor: Colors.white,
        child: Column(
          children: [
            Container(
              height: 200,
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(22),
              ),
            ),
            Container(
              height: 110,
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  mainAxisExtent: 200,
                  crossAxisSpacing: 14,
                  mainAxisSpacing: 14,
                ),
                itemCount: 4,
                itemBuilder: (_, __) => Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(18),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPopularMedicines(String selectedCategory) {
    final medicinesAsync = ref.watch(medicinesProvider(selectedCategory));

    return medicinesAsync.when(
      loading: () => SliverToBoxAdapter(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Shimmer.fromColors(
            baseColor: Colors.grey[300]!,
            highlightColor: Colors.grey[100]!,
            child: GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                mainAxisExtent: 200,
                crossAxisSpacing: 14,
                mainAxisSpacing: 14,
              ),
              itemCount: 4,
              itemBuilder: (_, __) => Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                ),
              ),
            ),
          ),
        ),
      ),
      error: (error, stackTrace) => SliverToBoxAdapter(
        child: ErrorStateWidget(
          message: 'Could not load medicines. Check connection.',
          onRetry: () => ref.invalidate(medicinesProvider(selectedCategory)),
        ),
      ),
      data: (medicines) {
        if (medicines.isEmpty) {
          return const SliverToBoxAdapter(
            child: EmptyStateWidget(
              emoji: '💊',
              title: 'No Medicines Found',
              subtitle: 'We couldn\'t find any medicines in this category.',
            ),
          );
        }
        final displayMedicines = medicines.take(6).toList();

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
                      'Popular Medicines',
                      style: GoogleFonts.sora(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary,
                      ),
                    ),
                    GestureDetector(
                      onTap: () => context.push('/search'),
                      child: Text(
                        'View all',
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
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final crossAxisCount = constraints.maxWidth < 360 ? 2 : 3;
                    final itemHeight =
                        constraints.maxWidth < 360 ? 200.0 : 220.0;

                    return GridView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: crossAxisCount,
                        mainAxisExtent: itemHeight,
                        crossAxisSpacing: 12,
                        mainAxisSpacing: 12,
                      ),
                      itemCount: displayMedicines.length,
                      itemBuilder: (context, index) {
                        final medicine = displayMedicines[index];
                        return HomeMedicineCard(
                          medicine: medicine.toMap(),
                          onTap: () => _openMedicine(medicine),
                          onAddToCart: () => _addToCartModel(medicine),
                        );
                      },
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildArticlesSection() {
    final articlesAsync = ref.watch(articlesProvider);

    return articlesAsync.when(
      loading: () => const HomeArticlesShimmer(),
      error: (_, __) => const SliverToBoxAdapter(child: SizedBox.shrink()),
      data: (articles) => HomeArticlesSection(articles: articles),
    );
  }
}

class _MedicineSearchDelegate extends SearchDelegate<void> {
  final WidgetRef ref;
  final Function(MedicineModel) onTapMedicine;

  _MedicineSearchDelegate({required this.ref, required this.onTapMedicine});

  @override
  String get searchFieldLabel => 'Search medicines...';

  @override
  ThemeData appBarTheme(BuildContext context) {
    return Theme.of(context).copyWith(
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
      ),
      inputDecorationTheme: const InputDecorationTheme(
        hintStyle: TextStyle(color: Colors.white70),
        border: InputBorder.none,
      ),
    );
  }

  @override
  List<Widget>? buildActions(BuildContext context) {
    return [
      IconButton(
        icon: const Icon(Icons.mic),
        onPressed: () {},
      ),
    ];
  }

  @override
  Widget? buildLeading(BuildContext context) {
    return IconButton(
      icon: const Icon(Icons.arrow_back),
      onPressed: () => close(context, null),
    );
  }

  @override
  Widget buildResults(BuildContext context) {
    return _buildSearchResults();
  }

  @override
  Widget buildSuggestions(BuildContext context) {
    return _buildSearchResults();
  }

  Widget _buildSearchResults() {
    if (query.isEmpty) {
      return const Center(
        child: Text('Start typing to search'),
      );
    }

    final allMedicinesAsync = ref.watch(allMedicinesProvider);

    return allMedicinesAsync.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (_, __) => const Center(child: Text('Error loading medicines')),
      data: (allMedicines) {
        final results = allMedicines
            .where((m) =>
                m.name.toLowerCase().contains(query.toLowerCase()) ||
                m.genericName.toLowerCase().contains(query.toLowerCase()) ||
                m.manufacturer.toLowerCase().contains(query.toLowerCase()))
            .toList();

        if (results.isEmpty) {
          return const Center(child: Text('No medicines found'));
        }

        return ListView.builder(
          itemCount: results.length,
          itemBuilder: (context, index) {
            final medicine = results[index];
            return ListTile(
              leading: Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: AppColors.primaryLight,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.medication, color: AppColors.primary),
              ),
              title: Text(medicine.name),
              subtitle: Text(medicine.manufacturer),
              trailing: Text(
                '\u20B9${medicine.price.toStringAsFixed(0)}',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: AppColors.primary,
                ),
              ),
              onTap: () {
                close(context, null);
                onTapMedicine(medicine);
              },
            );
          },
        );
      },
    );
  }
}
