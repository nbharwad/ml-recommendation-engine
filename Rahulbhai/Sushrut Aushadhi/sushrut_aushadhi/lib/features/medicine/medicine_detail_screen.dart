import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../core/constants/app_colors.dart';
import '../../core/utils/helpers.dart';
import '../../models/medicine_model.dart';
import '../../providers/cart_provider.dart';
import '../../providers/medicines_provider.dart';

class MedicineDetailScreen extends ConsumerStatefulWidget {
  final Map<String, dynamic>? medicine;
  final Map<String, dynamic>? localMedicine;
  final String? medicineId;

  const MedicineDetailScreen({
    super.key,
    this.medicine,
    this.localMedicine,
    this.medicineId,
  });

  @override
  ConsumerState<MedicineDetailScreen> createState() => _MedicineDetailScreenState();
}

class _MedicineDetailScreenState extends ConsumerState<MedicineDetailScreen> {
  int _quantity = 1;
  bool _isWishlisted = false;

  Map<String, dynamic>? _initialMedicine() {
    if (widget.localMedicine != null) {
      return Map<String, dynamic>.from(widget.localMedicine!);
    }
    if (widget.medicine != null) {
      return Map<String, dynamic>.from(widget.medicine!);
    }
    return null;
  }

  Map<String, dynamic> _medicineMap(MedicineModel medicine) {
    return {
      'id': medicine.id,
      'name': medicine.name,
      'genericName': medicine.genericName,
      'brand': medicine.manufacturer,
      'manufacturer': medicine.manufacturer,
      'category': medicine.category,
      'price': medicine.price,
      'mrp': medicine.mrp,
      'stock': medicine.stock,
      'unit': medicine.unit,
      'imageUrl': medicine.imageUrl,
      'requiresPrescription': medicine.requiresPrescription,
      'description': medicine.description,
    };
  }

  @override
  Widget build(BuildContext context) {
    final cartCount = ref.watch(cartItemCountProvider);
    final initialMedicine = _initialMedicine();
    final medicineId = widget.medicineId ?? initialMedicine?['id']?.toString();

    if (initialMedicine != null) {
      return Scaffold(
        backgroundColor: AppColors.background,
        body: _buildBody(context, cartCount, initialMedicine),
      );
    }

    if (medicineId == null || medicineId.isEmpty) {
      return const Scaffold(
        body: Center(child: Text('Medicine not found')),
      );
    }

    final medicineAsync = ref.watch(medicineByIdProvider(medicineId));
    return Scaffold(
      backgroundColor: AppColors.background,
      body: medicineAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, stackTrace) =>
            const Center(child: Text('Medicine not found')),
        data: (medicine) {
          if (medicine == null) {
            return const Center(child: Text('Medicine not found'));
          }
          return _buildBody(context, cartCount, _medicineMap(medicine));
        },
      ),
    );
  }

  Widget _buildBody(
    BuildContext context,
    int cartCount,
    Map<String, dynamic> medicine,
  ) {
    return Stack(
      children: [
        CustomScrollView(
          slivers: [
            _buildAppBar(context, cartCount),
            SliverToBoxAdapter(
              child: Column(
                children: [
                  _buildHeroSection(medicine),
                  _buildQuantityRow(),
                  _buildProductInfoSection(medicine),
                  _buildDescriptionSection(medicine),
                  _buildTrustSection(),
                  const SizedBox(height: 120),
                ],
              ),
            ),
          ],
        ),
        _buildStickyBottomBar(medicine),
      ],
    );
  }

  Widget _buildAppBar(BuildContext context, int cartCount) {
    return SliverAppBar(
      backgroundColor: Colors.white,
      elevation: 0,
      pinned: true,
      leading: Container(
        margin: const EdgeInsets.all(8),
        decoration: BoxDecoration(color: Colors.grey.shade200, shape: BoxShape.circle),
        child: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary, size: 20),
          onPressed: () => context.pop(),
        ),
      ),
      title: Text(
        'Medicine Detail',
        style: GoogleFonts.sora(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: AppColors.textPrimary,
        ),
      ),
      centerTitle: true,
      actions: [
        IconButton(
          icon: const Icon(Icons.share_outlined, color: AppColors.textPrimary, size: 20),
          onPressed: () {},
        ),
        Stack(
          children: [
            IconButton(
              icon: const Icon(Icons.shopping_cart_outlined, color: AppColors.textPrimary, size: 20),
              onPressed: () => context.push('/cart'),
            ),
            if (cartCount > 0)
              Positioned(
                right: 6,
                top: 6,
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: const BoxDecoration(color: AppColors.primary, shape: BoxShape.circle),
                  child: Text(
                    '$cartCount',
                    style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(width: 8),
      ],
    );
  }

  Widget _buildHeroSection(Map<String, dynamic> medicine) {
    final price = (medicine['price'] as num?)?.toDouble() ?? 0;
    final mrp = (medicine['mrp'] as num?)?.toDouble() ?? 0;
    final discount = Helpers.calculateDiscount(price, mrp);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        children: [
          Container(
            width: 110,
            height: 110,
            decoration: BoxDecoration(
              color: AppColors.primaryLight,
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Center(
              child: Icon(Icons.medication_rounded, size: 56, color: AppColors.primary),
            ),
          ),
          const SizedBox(height: 16),
          if (medicine['requiresPrescription'] == true)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppColors.error.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Rx Required',
                style: GoogleFonts.sora(
                  color: AppColors.error,
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                ),
              ),
            ),
          const SizedBox(height: 12),
          Text(
            medicine['name']?.toString() ?? '',
            style: GoogleFonts.sora(
              fontSize: 17,
              fontWeight: FontWeight.bold,
              color: AppColors.textPrimary,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
          Text(
            '${medicine['brand'] ?? ''} • ${medicine['unit'] ?? '1 strip'}',
            style: GoogleFonts.sora(fontSize: 12, color: AppColors.textSecondary),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              FittedBox(
                fit: BoxFit.scaleDown,
                child: Text(
                  '\u20B9${price.toStringAsFixed(0)}',
                  style: GoogleFonts.sora(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primary,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Flexible(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  child: Text(
                    '\u20B9${mrp.toStringAsFixed(0)}',
                    style: GoogleFonts.sora(
                      fontSize: 14,
                      color: AppColors.textSecondary,
                      decoration: TextDecoration.lineThrough,
                    ),
                  ),
                ),
              ),
              if (discount > 0) ...[
                const SizedBox(width: 8),
                Flexible(
                  child: FittedBox(
                    fit: BoxFit.scaleDown,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: AppColors.discountRed,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '$discount% off',
                        style: GoogleFonts.sora(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                          fontSize: 11,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildQuantityRow() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 320;
          final controls = Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildQuantityButton(
                icon: Icons.remove,
                onTap: _quantity > 1 ? () => setState(() => _quantity--) : null,
              ),
              SizedBox(
                width: 40,
                child: Text(
                  '$_quantity',
                  style: GoogleFonts.sora(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primary,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              _buildQuantityButton(
                icon: Icons.add,
                onTap: _quantity < 10 ? () => setState(() => _quantity++) : null,
              ),
            ],
          );

          if (compact) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Quantity',
                  style: GoogleFonts.sora(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 12),
                controls,
              ],
            );
          }

          return Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Quantity',
                style: GoogleFonts.sora(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
              controls,
            ],
          );
        },
      ),
    );
  }

  Widget _buildQuantityButton({required IconData icon, VoidCallback? onTap}) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          border: Border.all(color: AppColors.primary),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(
          icon,
          color: onTap == null ? AppColors.textSecondary : AppColors.primary,
          size: 16,
        ),
      ),
    );
  }

  Widget _buildProductInfoSection(Map<String, dynamic> medicine) {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Product Information',
            style: GoogleFonts.sora(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 12),
          _buildInfoRow('Generic Name', (medicine['genericName'] ?? medicine['name'] ?? '-').toString()),
          const Divider(height: 1),
          _buildInfoRow('Manufacturer', (medicine['brand'] ?? medicine['manufacturer'] ?? '-').toString()),
          const Divider(height: 1),
          _buildInfoRow('Category', (medicine['category'] ?? '-').toString()),
          const Divider(height: 1),
          _buildInfoRow('Pack of', (medicine['unit'] ?? '1 strip').toString()),
          const Divider(height: 1),
          _buildInfoRow('Storage', 'Store below 30°C'),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 280;
          if (compact) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.sora(fontSize: 12, color: AppColors.textSecondary),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: GoogleFonts.sora(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
              ],
            );
          }
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                width: 100,
                child: Text(
                  label,
                  style: GoogleFonts.sora(fontSize: 12, color: AppColors.textSecondary),
                ),
              ),
              Expanded(
                child: Text(
                  value,
                  style: GoogleFonts.sora(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildDescriptionSection(Map<String, dynamic> medicine) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Description',
            style: GoogleFonts.sora(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            medicine['description']?.toString() ?? 'No description available.',
            style: GoogleFonts.sora(
              fontSize: 12,
              color: AppColors.textSecondary,
              height: 1.7,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrustSection() {
    return Container(
      margin: const EdgeInsets.fromLTRB(14, 8, 14, 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFEDF2ED)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Why buy from us?',
            style: GoogleFonts.sora(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 10),
          _whyUsRow('✅', 'Genuine medicines only', '100% authentic, sourced from licensed distributors'),
          _whyUsRow('👨‍⚕️', 'Pharmacist verified', 'Every order checked by registered pharmacist'),
          _whyUsRow('🚚', 'Fast delivery', 'Delivered within 2 hours in Bengaluru'),
          _whyUsRow('↩️', 'Easy returns', 'Hassle-free return for damaged medicines'),
        ],
      ),
    );
  }

  Widget _whyUsRow(String emoji, String title, String subtitle) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(emoji, style: const TextStyle(fontSize: 16)),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.sora(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
                Text(
                  subtitle,
                  style: GoogleFonts.sora(
                    fontSize: 11,
                    color: AppColors.textSecondary,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStickyBottomBar(Map<String, dynamic> medicine) {
    final totalPrice = ((medicine['price'] as num?)?.toDouble() ?? 0) * _quantity;

    return Positioned(
      bottom: 0,
      left: 0,
      right: 0,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.grey.shade300,
              blurRadius: 10,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        child: SafeArea(
          top: false,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final compact = constraints.maxWidth < 340;
              final actionButton = SizedBox(
                width: compact ? double.infinity : null,
                child: ElevatedButton(
                  onPressed: () => _addToCart(medicine),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    elevation: 0,
                  ),
                  child: Text(
                    'Add to Cart • \u20B9${totalPrice.toStringAsFixed(0)}',
                    style: GoogleFonts.sora(fontWeight: FontWeight.w600, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                ),
              );

              final wishlist = InkWell(
                onTap: () => setState(() => _isWishlisted = !_isWishlisted),
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: _isWishlisted ? AppColors.error : AppColors.primary,
                    ),
                    borderRadius: BorderRadius.circular(12),
                    color: _isWishlisted ? AppColors.error.withOpacity(0.1) : Colors.transparent,
                  ),
                  child: Icon(
                    _isWishlisted ? Icons.favorite : Icons.favorite_border,
                    color: _isWishlisted ? AppColors.error : AppColors.primary,
                    size: 20,
                  ),
                ),
              );

              if (compact) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Align(alignment: Alignment.centerLeft, child: wishlist),
                    const SizedBox(height: 12),
                    actionButton,
                  ],
                );
              }

              return Row(
                children: [
                  wishlist,
                  const SizedBox(width: 12),
                  Expanded(child: actionButton),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  void _addToCart(Map<String, dynamic> medicine) {
    if (medicine['requiresPrescription'] == true) {
      showModalBottomSheet(
        context: context,
        isScrollControlled: true,
        builder: (context) => SafeArea(
          child: Padding(
            padding: EdgeInsets.fromLTRB(24, 24, 24, 24 + MediaQuery.of(context).viewInsets.bottom),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.warning_amber_rounded, color: AppColors.accent, size: 48),
                const SizedBox(height: 16),
                Text(
                  'Prescription Required',
                  style: GoogleFonts.sora(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  'This medicine needs a prescription. Please upload prescription when you place your order.',
                  style: GoogleFonts.sora(fontSize: 14, color: AppColors.textSecondary),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                      _proceedToAddToCart(medicine);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      elevation: 0,
                    ),
                    child: Text('OK, Got it', style: GoogleFonts.sora(fontWeight: FontWeight.w600)),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
      return;
    }

    _proceedToAddToCart(medicine);
  }

  void _proceedToAddToCart(Map<String, dynamic> medicine) {
    final cartNotifier = ref.read(cartProvider.notifier);
    final medicineModel = MedicineModel(
      id: medicine['id']?.toString() ?? '',
      name: medicine['name']?.toString() ?? '',
      genericName: medicine['genericName']?.toString() ?? '',
      manufacturer:
          medicine['manufacturer']?.toString() ?? medicine['brand']?.toString() ?? '',
      description: medicine['description']?.toString() ?? '',
      price: (medicine['price'] as num?)?.toDouble() ?? 0,
      mrp: (medicine['mrp'] as num?)?.toDouble() ?? 0,
      imageUrl: medicine['imageUrl']?.toString() ?? '',
      category: medicine['category']?.toString() ?? 'other',
      requiresPrescription: medicine['requiresPrescription'] == true,
      stock: (medicine['stock'] as num?)?.toInt() ?? 100,
      unit: medicine['unit']?.toString() ?? 'strip',
    );

    cartNotifier.addItem(medicineModel, quantity: _quantity);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('${medicine['name']} added to cart'),
        backgroundColor: AppColors.primary,
        action: SnackBarAction(
          label: 'View Cart',
          textColor: Colors.white,
          onPressed: () => context.push('/cart'),
        ),
      ),
    );
  }
}
