import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../core/constants/app_colors.dart';
import '../../core/widgets/search_header_widget.dart';
import '../../core/widgets/search_suggestions_widget.dart';
import '../../core/widgets/search_result_card.dart';
import '../../core/widgets/search_empty_state.dart';
import '../../models/medicine_model.dart';
import '../../providers/cart_provider.dart';
import '../../providers/search_provider.dart';
import '../../core/di/service_providers.dart';
import '../../services/search_history_service.dart';

class SearchScreen extends ConsumerStatefulWidget {
  const SearchScreen({super.key});

  @override
  ConsumerState<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends ConsumerState<SearchScreen> {
  final _controller = TextEditingController();
  final _focusNode = FocusNode();
  String _query = '';
  List<String> _searchHistory = [];
  bool _showSuggestions = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _focusNode.requestFocus();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  Future<void> _loadHistory() async {
    final history = await SearchHistoryService.getHistory();
    setState(() => _searchHistory = history);
  }

  void _onSearchChanged(String val) {
    Future.delayed(const Duration(milliseconds: 300), () {
      if (val == _controller.text && val.trim().length >= 2) {
        setState(() {
          _query = val.trim();
          _showSuggestions = false;
        });
      } else if (val.trim().isEmpty) {
        setState(() {
          _query = '';
          _showSuggestions = true;
        });
      }
    });
  }

  void _onSubmitted(String val) {
    if (val.trim().isNotEmpty) {
      _saveSearch(val.trim());
    }
  }

  void _onClear() {
    _controller.clear();
    setState(() {
      _query = '';
      _showSuggestions = true;
    });
  }

  Future<void> _saveSearch(String query) async {
    await SearchHistoryService.addSearch(query);
    _loadHistory();
  }

  void _onHistoryTap(String query) {
    _controller.text = query;
    setState(() {
      _query = query;
      _showSuggestions = false;
    });
    _saveSearch(query);
  }

  void _onPopularTap(String query) {
    _controller.text = query;
    setState(() {
      _query = query;
      _showSuggestions = false;
    });
    _saveSearch(query);
  }

  Future<void> _clearHistory() async {
    await SearchHistoryService.clearHistory();
    setState(() => _searchHistory = []);
  }

  void _addToCart(MedicineModel medicine) {
    ref.read(cartProvider.notifier).addItem(medicine);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: AppColors.primary,
        content:
            Text('${medicine.name} added to cart', style: GoogleFonts.sora()),
      ),
    );
  }

  Future<void> _openMedicine(MedicineModel medicine) async {
    await _saveSearch(_query);
    final full =
        await ref.read(firestoreServiceProvider).getMedicineById(medicine.id);
    if (full != null && mounted) {
      context.push('/medicine', extra: full.toMap());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: Column(
        children: [
          SearchHeaderWidget(
            controller: _controller,
            focusNode: _focusNode,
            onChanged: _onSearchChanged,
            onSubmitted: _onSubmitted,
            onClear: _onClear,
            onBack: () => context.pop(),
          ),
          Expanded(
            child: _showSuggestions
                ? SearchSuggestionsWidget(
                    recentSearches: _searchHistory,
                    onHistoryTap: _onHistoryTap,
                    onClearHistory: _clearHistory,
                    onPopularTap: _onPopularTap,
                  )
                : _buildResults(),
          ),
        ],
      ),
    );
  }

  Widget _buildResults() {
    if (_query.isEmpty) {
      return SearchSuggestionsWidget(
        recentSearches: _searchHistory,
        onHistoryTap: _onHistoryTap,
        onClearHistory: _clearHistory,
        onPopularTap: _onPopularTap,
      );
    }

    final resultsAsync = ref.watch(hybridSearchProvider(_query));

    return resultsAsync.when(
      loading: () => const SearchLoadingState(),
      error: (error, stackTrace) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline,
                size: 48, color: AppColors.textSecondary),
            const SizedBox(height: 16),
            Text(
              'Could not search medicines.\nCheck connection.',
              style: GoogleFonts.sora(
                  fontSize: 14, color: AppColors.textSecondary),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
      data: (items) {
        if (items.isEmpty) {
          return SearchEmptyState(query: _query);
        }
        return SearchResultsList(
          medicines: items,
          query: _query,
          onMedicineTap: _openMedicine,
          onAddToCart: _addToCart,
          onRefresh: () => setState(() {}),
        );
      },
    );
  }
}
