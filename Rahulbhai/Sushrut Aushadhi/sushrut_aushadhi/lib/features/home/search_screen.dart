import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../models/medicine_model.dart';
import '../../providers/cart_provider.dart';
import '../../providers/search_provider.dart';
import '../../core/di/service_providers.dart';
import '../../services/search_history_service.dart';
import '../../core/utils/helpers.dart';

const _primary = Color(0xFF0F6E56);
const _primaryLight = Color(0xFFE1F5EE);
const _background = Color(0xFFF7F9F7);
const _textPrimary = Color(0xFF1A1A1A);
const _textSecondary = Color(0xFF666666);
const _discountRed = Color(0xFFE53935);

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

  Future<void> _saveSearch(String query) async {
    await SearchHistoryService.addSearch(query);
    _loadHistory();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _background,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        surfaceTintColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded, color: _textPrimary),
          onPressed: () => context.pop(),
        ),
        title: TextField(
          controller: _controller,
          focusNode: _focusNode,
          onChanged: _onSearchChanged,
          onSubmitted: (val) {
            if (val.trim().isNotEmpty) {
              _saveSearch(val.trim());
            }
          },
          style: GoogleFonts.sora(fontSize: 14, color: _textPrimary),
          decoration: InputDecoration(
            hintText: 'Search medicines, health products...',
            hintStyle: GoogleFonts.sora(fontSize: 14, color: _textSecondary),
            border: InputBorder.none,
            contentPadding: EdgeInsets.zero,
          ),
        ),
        actions: [
          if (_controller.text.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.close_rounded, color: _textSecondary),
              onPressed: () {
                _controller.clear();
                setState(() {
                  _query = '';
                  _showSuggestions = true;
                });
              },
            ),
        ],
      ),
      body: _showSuggestions ? _buildSuggestions() : _buildResults(),
    );
  }

  Widget _buildSuggestions() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_searchHistory.isNotEmpty) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Recent Searches',
                  style: GoogleFonts.sora(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: _textPrimary,
                  ),
                ),
                GestureDetector(
                  onTap: () async {
                    await SearchHistoryService.clearHistory();
                    setState(() => _searchHistory = []);
                  },
                  child: Text(
                    'Clear all',
                    style: GoogleFonts.sora(
                      fontSize: 11,
                      color: _primary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            ..._searchHistory.map((query) => Container(
                  margin: const EdgeInsets.only(bottom: 6),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0xFFEDF2ED)),
                  ),
                  child: ListTile(
                    dense: true,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 0),
                    leading: const Icon(
                      Icons.history,
                      size: 18,
                      color: Colors.grey,
                    ),
                    title: Text(
                      query,
                      style: GoogleFonts.sora(
                        fontSize: 13,
                        color: _textPrimary,
                      ),
                    ),
                    trailing: GestureDetector(
                      onTap: () async {
                        await SearchHistoryService.removeSearch(query);
                        _loadHistory();
                      },
                      child: const Icon(
                        Icons.close,
                        size: 16,
                        color: Colors.grey,
                      ),
                    ),
                    onTap: () {
                      _controller.text = query;
                      setState(() => _query = query);
                      _saveSearch(query);
                    },
                  ),
                )),
            const SizedBox(height: 20),
          ],
          Text(
            'Popular Searches',
            style: GoogleFonts.sora(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: _textPrimary,
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              'Paracetamol',
              'Vitamin D',
              'Amoxicillin',
              'Ibuprofen',
              'Metformin',
              'Cetirizine',
              'Omeprazole',
              'Azithromycin',
            ].map((s) => GestureDetector(
                  onTap: () {
                    _controller.text = s;
                    setState(() => _query = s);
                    _saveSearch(s);
                  },
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                      color: _primaryLight,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: const Color(0xFF9FE1CB)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Text('🔍', style: TextStyle(fontSize: 11)),
                        const SizedBox(width: 4),
                        Text(
                          s,
                          style: GoogleFonts.sora(
                            fontSize: 12,
                            color: _primary,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                )).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildResults() {
    if (_query.isEmpty) {
      return _buildSuggestions();
    }

    final resultsAsync = ref.watch(hybridSearchProvider(_query));

    return resultsAsync.when(
      loading: () => const Center(
        child: CircularProgressIndicator(color: _primary),
      ),
      error: (error, stackTrace) => Center(
        child: Text(
          'Could not search medicines. Check connection.',
          style: GoogleFonts.sora(fontSize: 14, color: _textSecondary),
        ),
      ),
      data: (items) {
        if (items.isEmpty) {
          return _buildEmptyState();
        }
        return _buildResultsList(items);
      },
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text('🔍', style: TextStyle(fontSize: 48)),
          const SizedBox(height: 16),
          Text(
            'No Results Found',
            style: GoogleFonts.sora(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: _textPrimary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'We couldn\'t find any medicines matching "$_query".',
            style: GoogleFonts.sora(
              fontSize: 14,
              color: _textSecondary,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildResultsList(List<MedicineModel> items) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: Text(
            '${items.length} result${items.length == 1 ? '' : 's'} for "$_query"',
            style: GoogleFonts.sora(
              fontSize: 12,
              color: _textSecondary,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        Expanded(
          child: ListView.separated(
            padding: const EdgeInsets.all(12),
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 0.5),
            itemBuilder: (ctx, i) {
              final medicine = items[i];
              final discount = Helpers.calculateDiscount(medicine.price, medicine.mrp);
              return InkWell(
                onTap: () async {
                  await _saveSearch(_query);
                  final full = await ref.read(firestoreServiceProvider).getMedicineById(medicine.id);
                  if (full != null && mounted) {
                    context.push('/medicine', extra: full.toMap());
                  }
                },
                borderRadius: BorderRadius.circular(16),
                child: Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: const Color(0xFFE8ECE7)),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 66,
                        height: 66,
                        decoration: BoxDecoration(
                          color: const Color(0xFFF3F5F2),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Center(
                          child: Icon(
                            _searchResultIcon(medicine.category),
                            size: 32,
                            color: _primary,
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
                                  child: Text(
                                    medicine.name,
                                    style: GoogleFonts.sora(
                                      fontSize: 14,
                                      color: _textPrimary,
                                      fontWeight: FontWeight.w700,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                                if (medicine.requiresPrescription)
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 6, vertical: 3),
                                    decoration: BoxDecoration(
                                      color: const Color(0xFFFFEBEE),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Text(
                                      'Rx',
                                      style: GoogleFonts.sora(
                                        fontSize: 10,
                                        color: _discountRed,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                            const SizedBox(height: 4),
                            Text(
                              medicine.manufacturer,
                              style: GoogleFonts.sora(
                                fontSize: 11,
                                color: _textSecondary,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Wrap(
                              spacing: 6,
                              children: [
                                Text(
                                  '\u20B9${medicine.price.toStringAsFixed(0)}',
                                  style: GoogleFonts.sora(
                                    fontSize: 14,
                                    color: _primary,
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                                Text(
                                  '\u20B9${medicine.mrp.toStringAsFixed(0)}',
                                  style: GoogleFonts.sora(
                                    fontSize: 11,
                                    color: _textSecondary,
                                    decoration: TextDecoration.lineThrough,
                                  ),
                                ),
                                if (discount > 0)
                                  Text(
                                    '$discount% off',
                                    style: GoogleFonts.sora(
                                      fontSize: 11,
                                      color: _discountRed,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      ElevatedButton(
                        onPressed: () {
                          ref.read(cartProvider.notifier).addItem(medicine);
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              backgroundColor: _primary,
                              content: Text(
                                '${medicine.name} added to cart',
                                style: GoogleFonts.sora(),
                              ),
                            ),
                          );
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _primary,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 10),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        child: Text(
                          'Add',
                          style: GoogleFonts.sora(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
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
