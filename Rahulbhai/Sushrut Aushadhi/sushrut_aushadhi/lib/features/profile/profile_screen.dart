import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_strings.dart';
import '../../core/utils/responsive.dart';
import '../../services/remote_config_service.dart';
import '../../core/widgets/menu_item_tile.dart';
import '../../core/di/service_providers.dart';
import '../../models/lab_order_model.dart';
import '../../models/order_model.dart';
import '../../models/user_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/cart_provider.dart';
import '../../providers/lab_providers.dart';
import '../../providers/orders_provider.dart';

class ProfileScreen extends ConsumerStatefulWidget {
  final String? redirectTo;

  const ProfileScreen({super.key, this.redirectTo});

  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  void _testCrash() {
    if (kDebugMode) {
      FirebaseCrashlytics.instance.crash();
    }
  }

  Future<void> _logout() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Logout', style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
        content: Text(
          'Are you sure you want to logout?',
          style: GoogleFonts.sora(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: Text('Cancel', style: GoogleFonts.sora(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: Text('Logout', style: GoogleFonts.sora(color: AppColors.error)),
          ),
        ],
      ),
    );

    if (confirm == true && mounted) {
      await ref.read(authServiceProvider).signOut();
      context.go('/home');
    }
  }

  void _showEditProfileSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => const EditProfileBottomSheet(),
    );
  }

  void _showAddressSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => AddressBottomSheet(redirectTo: widget.redirectTo),
    );
  }

  @override
  Widget build(BuildContext context) {
    final currentUser = ref.watch(currentUserProvider);
    final cartItemCount = ref.watch(cartItemCountProvider);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: currentUser.when(
        data: (user) => user == null
            ? _buildGuestView(cartItemCount)
            : _buildUserView(user, cartItemCount),
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(
          child: Text('Error: $error', style: GoogleFonts.sora()),
        ),
      ),
    );
  }

  Widget _buildGuestView(int cartItemCount) {
    return Column(
      children: [
        _buildHeroHeader(null),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.only(bottom: 24),
            child: Column(
              children: [
                const SizedBox(height: 16),
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 16),
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    children: [
                      Text(
                        'Login to view your orders and profile',
                        style: GoogleFonts.sora(
                          color: AppColors.textSecondary,
                          fontSize: 14,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 16),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: () => context.go('/login'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primary,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          child: Text(
                            'Login / Sign Up',
                            style: GoogleFonts.sora(
                              fontWeight: FontWeight.w600,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildUserView(UserModel user, int cartItemCount) {
    final isAdmin = ref.watch(isAdminProvider);
    final allOrders = ref.watch(ordersProvider).valueOrNull ?? [];
    final activeCount = allOrders.where((o) =>
      o.status == OrderStatus.pending ||
      o.status == OrderStatus.confirmed ||
      o.status == OrderStatus.preparing ||
      o.status == OrderStatus.outForDelivery
    ).length;
    final totalOrders = allOrders.length;
    final allLabOrders = ref.watch(userLabOrdersProvider).valueOrNull ?? [];
    final activeLabCount = allLabOrders.where((o) =>
      o.status == LabOrderStatus.pending ||
      o.status == LabOrderStatus.sampleCollected ||
      o.status == LabOrderStatus.processing
    ).length;
    
    return Column(
      children: [
        _buildHeroHeader(user, totalOrders: totalOrders),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.only(bottom: 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (isAdmin) ...[
                  const SizedBox(height: 8),
                  _buildSectionTitle('ADMIN'),
                  _buildMenuSection([
                    MenuItemTile(
                      icon: const Icon(Icons.admin_panel_settings, color: AppColors.primary, size: 18),
                      iconBg: const Color(0xFFE1F5EE),
                      title: 'Admin Orders',
                      subtitle: 'Manage customer orders',
                      onTap: () => context.push('/admin'),
                    ),
                    MenuItemTile(
                      icon: const Icon(Icons.medication, color: Color(0xFF8E24AA), size: 18),
                      iconBg: const Color(0xFFF3E5F5),
                      title: 'Prescriptions',
                      subtitle: 'Review prescriptions',
                      onTap: () => context.push('/admin/prescriptions'),
                    ),
                  ]),
                  const SizedBox(height: 8),
                ],
                const SizedBox(height: 8),
                _buildSectionTitle('ACCOUNT'),
                _buildMenuSection([
                  MenuItemTile(
                    icon: const Icon(Icons.receipt_long, color: AppColors.primary, size: 18),
                    iconBg: const Color(0xFFE1F5EE),
                    title: 'My Orders',
                    subtitle: 'Track & manage your orders',
                    trailing: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: AppColors.error.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '$activeCount active',
                        style: GoogleFonts.sora(
                          color: AppColors.error,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    onTap: () => context.go('/orders'),
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.biotech, color: AppColors.labPrimary, size: 18),
                    iconBg: AppColors.labPrimaryLight,
                    title: 'My Lab Tests',
                    subtitle: 'View & track lab bookings',
                    trailing: activeLabCount > 0
                        ? Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: AppColors.labPrimary.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              '$activeLabCount active',
                              style: GoogleFonts.sora(
                                color: AppColors.labPrimary,
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          )
                        : null,
                    onTap: () => context.push('/lab/orders'),
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.location_on_outlined, color: Color(0xFF1E88E5), size: 18),
                    iconBg: const Color(0xFFE3F2FD),
                    title: 'Saved Addresses',
                    subtitle: 'Home, Work & more',
                    onTap: _showAddressSheet,
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.description_outlined, color: Color(0xFF8E24AA), size: 18),
                    iconBg: const Color(0xFFF3E5F5),
                    title: 'My Prescriptions',
                    subtitle: 'Medicine prescriptions',
                    onTap: () => context.push('/my-prescriptions'),
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.assignment_outlined, color: AppColors.labPrimary, size: 18),
                    iconBg: AppColors.labPrimaryLight,
                    title: 'Lab Prescriptions',
                    subtitle: 'Uploaded lab prescriptions',
                    onTap: () => context.push('/my-prescriptions?type=lab'),
                  ),
                ]),
                const SizedBox(height: 8),
                _buildSectionTitle('SUPPORT'),
                _buildMenuSection([
                  MenuItemTile(
                    icon: const Icon(Icons.chat_bubble_outline, color: Color(0xFFFB8C00), size: 18),
                    iconBg: const Color(0xFFFFF3E0),
                    title: 'Chat with Us',
                    subtitle: 'WhatsApp support',
                    onTap: () async {
                      final message = Uri.encodeComponent(
                        'Hello Sushrut Aushadhi! I need help with my order.'
                      );
                      final url = 'https://wa.me/${RemoteConfigService.storePhone}?text=$message';
                      await _launchUrl(url);
                    },
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.call_outlined, color: Color(0xFF1E88E5), size: 18),
                    iconBg: const Color(0xFFE3F2FD),
                    title: 'Call Store',
                    subtitle: '+91 80 XXXX XXXX',
                    onTap: () => _launchUrl('tel:+918000000000'),
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.star_outline, color: AppColors.primary, size: 18),
                    iconBg: const Color(0xFFE1F5EE),
                    title: 'Rate the App',
                    subtitle: 'Share your feedback',
                    onTap: () => _launchUrl('https://play.google.com/store/apps'),
                  ),
                ]),
                const SizedBox(height: 8),
                _buildSectionTitle('LEGAL'),
                _buildMenuSection([
                  MenuItemTile(
                    icon: const Text('🏥', style: TextStyle(fontSize: 18)),
                    iconBg: const Color(0xFFE1F5EE),
                    title: 'About Sushrut Aushadhi',
                    subtitle: 'Our story, license & contact',
                    onTap: () => _showAboutDialog(context),
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.lock_outline, color: AppColors.primary, size: 18),
                    iconBg: const Color(0xFFE1F5EE),
                    title: 'Privacy Policy',
                    onTap: () => _launchUrl('https://sushrutaushadhi.com/privacy'),
                  ),
                  MenuItemTile(
                    icon: const Icon(Icons.description_outlined, color: AppColors.primary, size: 18),
                    iconBg: const Color(0xFFE1F5EE),
                    title: 'Terms of Service',
                    onTap: () => _launchUrl('https://sushrutaushadhi.com/terms'),
                  ),
                ]),
                const SizedBox(height: 16),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _logout,
                      icon: const Icon(Icons.logout),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFFFEBEE),
                        foregroundColor: AppColors.error,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      label: Text(
                        'Logout',
                        style: GoogleFonts.sora(
                          fontWeight: FontWeight.w600,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                Center(
                  child: GestureDetector(
                    onTap: _testCrash,
                    child: Text(
                      'Sushrut Aushadhi v1.0.0',
                      style: GoogleFonts.sora(
                        color: AppColors.textSecondary,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 100),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildHeroHeader(UserModel? user, {int totalOrders = 0}) {
    final isCompact = context.isCompactWidth;
    final name = user?.name ?? 'Guest';
    final initials =
        name.split(' ').map((e) => e.isNotEmpty ? e[0] : '').take(2).join().toUpperCase();
    final phone = user?.phone ?? '';

    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF0F6E56), Color(0xFF1D9E75)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            top: -30,
            right: -30,
            child: Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withOpacity(0.07),
              ),
            ),
          ),
          Positioned(
            bottom: -20,
            left: -20,
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withOpacity(0.05),
              ),
            ),
          ),
          SafeArea(
            bottom: false,
            child: Padding(
              padding: EdgeInsets.fromLTRB(
                isCompact ? 16 : 24,
                isCompact ? 16 : 24,
                isCompact ? 16 : 24,
                20,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        width: isCompact ? 52 : 60,
                        height: isCompact ? 52 : 60,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: Colors.white.withOpacity(0.4),
                            width: 2.5,
                          ),
                        ),
                        child: Center(
                          child: Text(
                            initials.isNotEmpty ? initials : 'GU',
                            style: GoogleFonts.sora(
                              color: Colors.white,
                              fontSize: isCompact ? 18 : 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              name,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.sora(
                                color: Colors.white,
                                fontSize: isCompact ? 15 : 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            if (phone.isNotEmpty) ...[
                              const SizedBox(height: 4),
                              Text(
                                phone,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.sora(
                                  color: Colors.white.withOpacity(0.7),
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                      if (user != null)
                        TextButton(
                          onPressed: _showEditProfileSheet,
                          style: TextButton.styleFrom(
                            foregroundColor: Colors.white,
                            backgroundColor: Colors.white.withOpacity(0.2),
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                          child: Text('Edit', style: GoogleFonts.sora(fontSize: 12)),
                        ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  LayoutBuilder(
                    builder: (context, constraints) {
                      final useWrap = constraints.maxWidth < 360;
                      final itemWidth = useWrap ? (constraints.maxWidth - 12) / 2 : constraints.maxWidth / 3 - 8;
                      return Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: [
                          _buildStatCard('Total Orders', '$totalOrders', Icons.shopping_bag, itemWidth),
                          _buildStatCard('Total Saved', '\u20B90', Icons.savings, itemWidth),
                          _buildStatCard('Rating', '4.8', Icons.star, itemWidth),
                        ],
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatCard(String label, String value, IconData icon, double width) {
    return SizedBox(
      width: width,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.15),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          children: [
            Icon(icon, color: Colors.white, size: 16),
            const SizedBox(height: 4),
            Text(
              value,
              style: GoogleFonts.sora(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              label,
              textAlign: TextAlign.center,
              style: GoogleFonts.sora(
                color: Colors.white.withOpacity(0.7),
                fontSize: 9,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Text(
        title,
        style: GoogleFonts.sora(
          color: AppColors.textSecondary,
          fontSize: 12,
          fontWeight: FontWeight.w600,
          letterSpacing: 1,
        ),
      ),
    );
  }

  Widget _buildMenuSection(List<Widget> children) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(children: children),
    );
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        contentPadding: const EdgeInsets.all(20),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('💊', style: TextStyle(fontSize: 48)),
            const SizedBox(height: 8),
            Text(
              RemoteConfigService.storeName,
              style: GoogleFonts.sora(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: AppColors.primary,
              ),
            ),
            Text(
              AppStrings.appTagline,
              style: GoogleFonts.sora(
                fontSize: 12,
                color: AppColors.textSecondary,
              ),
            ),
            const SizedBox(height: 16),
            const Divider(),
            const SizedBox(height: 8),
            _aboutRow('📍', 'Address', RemoteConfigService.storeAddress),
            _aboutRow('📞', 'Phone', RemoteConfigService.storePhone),
            _aboutRow('📋', 'Drug License', RemoteConfigService.drugLicenseNo),
            _aboutRow('🏛️', 'GST No.', RemoteConfigService.gstNumber),
            _aboutRow('📱', 'App Version', 'v${AppStrings.appVersion}'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(
              'Close',
              style: GoogleFonts.sora(
                color: AppColors.primary,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _aboutRow(String emoji, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(emoji, style: const TextStyle(fontSize: 14)),
          const SizedBox(width: 8),
          Text(
            '$label: ',
            style: GoogleFonts.sora(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.textSecondary,
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: GoogleFonts.sora(
                fontSize: 12,
                color: AppColors.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class EditProfileBottomSheet extends ConsumerStatefulWidget {
  const EditProfileBottomSheet({super.key});

  @override
  ConsumerState<EditProfileBottomSheet> createState() => _EditProfileBottomSheetState();
}

class _EditProfileBottomSheetState extends ConsumerState<EditProfileBottomSheet> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _nameController;
  late TextEditingController _addressController;
  late TextEditingController _pincodeController;
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController();
    _addressController = TextEditingController();
    _pincodeController = TextEditingController();

    ref.read(currentUserProvider).whenData((user) {
      if (user != null) {
        _nameController.text = user.name;
        _addressController.text = user.address;
        _pincodeController.text = user.pincode;
      }
    });
  }

  @override
  void dispose() {
    _nameController.dispose();
    _addressController.dispose();
    _pincodeController.dispose();
    super.dispose();
  }

  Future<void> _saveProfile() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() => _isSaving = true);

    try {
      final user = ref.read(currentUserProvider).value;
      if (user == null) {
        throw Exception('User not found');
      }

      await ref.read(firestoreServiceProvider).updateUser(user.uid, {
        'name': _nameController.text.trim(),
        'address': _addressController.text.trim(),
        'pincode': _pincodeController.text.trim(),
      });

      ref.invalidate(currentUserProvider);

      if (!mounted) {
        return;
      }
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Profile updated successfully', style: GoogleFonts.sora())),
      );
    } catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e', style: GoogleFonts.sora())),
      );
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(24, 24, 24, 24 + MediaQuery.of(context).padding.bottom),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text('Edit Profile', style: GoogleFonts.sora(fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 20),
              TextFormField(
                controller: _nameController,
                decoration: InputDecoration(
                  labelText: 'Name',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                ),
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _addressController,
                decoration: InputDecoration(
                  labelText: 'Address',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                ),
                maxLines: 3,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _pincodeController,
                decoration: InputDecoration(
                  labelText: 'Pincode',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                ),
                keyboardType: TextInputType.number,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _isSaving ? null : _saveProfile,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: _isSaving
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : Text(
                          'Save',
                          style: GoogleFonts.sora(fontWeight: FontWeight.w600, fontSize: 16),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class AddressBottomSheet extends ConsumerStatefulWidget {
  final String? redirectTo;

  const AddressBottomSheet({super.key, this.redirectTo});

  @override
  ConsumerState<AddressBottomSheet> createState() => _AddressBottomSheetState();
}

class _AddressBottomSheetState extends ConsumerState<AddressBottomSheet> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _phoneController;
  late TextEditingController _addressController;
  late TextEditingController _pincodeController;
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _phoneController = TextEditingController();
    _addressController = TextEditingController();
    _pincodeController = TextEditingController();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      final user = ref.read(currentUserProvider).valueOrNull;
      if (user != null && mounted) {
        _phoneController.text = user.phone;
        _addressController.text = user.address;
        _pincodeController.text = user.pincode;
      }
    });
  }

  @override
  void dispose() {
    _phoneController.dispose();
    _addressController.dispose();
    _pincodeController.dispose();
    super.dispose();
  }

  Future<void> _saveAddress() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() => _isSaving = true);

    try {
      final user = ref.read(currentUserProvider).value;
      if (user == null) {
        throw Exception('User not found');
      }

      await ref.read(firestoreServiceProvider).updateUser(user.uid, {
        'phone': _phoneController.text.trim(),
        'address': _addressController.text.trim(),
        'pincode': _pincodeController.text.trim(),
      });

      ref.invalidate(currentUserProvider);

      if (!mounted) {
        return;
      }
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Address saved successfully', style: GoogleFonts.sora())),
      );

      if (widget.redirectTo == 'cart') {
        await Future.delayed(const Duration(milliseconds: 300));
        if (mounted) {
          context.go('/cart?fromProfile=true');
        }
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e', style: GoogleFonts.sora())),
      );
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(24, 24, 24, 24 + MediaQuery.of(context).padding.bottom),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text('Saved Addresses', style: GoogleFonts.sora(fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 20),
              TextFormField(
                controller: _phoneController,
                decoration: InputDecoration(
                  labelText: 'Phone Number',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                ),
                keyboardType: TextInputType.phone,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _addressController,
                decoration: InputDecoration(
                  labelText: 'Address',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                ),
                maxLines: 3,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _pincodeController,
                decoration: InputDecoration(
                  labelText: 'Pincode',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                ),
                keyboardType: TextInputType.number,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _isSaving ? null : _saveAddress,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: _isSaving
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : Text(
                          'Save Address',
                          style: GoogleFonts.sora(fontWeight: FontWeight.w600, fontSize: 16),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
