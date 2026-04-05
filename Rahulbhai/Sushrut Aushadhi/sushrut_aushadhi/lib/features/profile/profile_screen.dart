import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/app_strings.dart';
import '../../core/widgets/profile_header_widget.dart';
import '../../core/widgets/profile_menu_section.dart';
import '../../services/remote_config_service.dart';
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
        title: Text('Logout',
            style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
        content: Text(
          'Are you sure you want to logout?',
          style: GoogleFonts.sora(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: Text('Cancel',
                style: GoogleFonts.sora(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            child:
                Text('Logout', style: GoogleFonts.sora(color: AppColors.error)),
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
    return CustomScrollView(
      slivers: [
        ProfileHeaderGuest(
          onLoginTap: () => context.go('/login'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Text(
                  'Login to view your orders and manage your profile',
                  style: GoogleFonts.sora(
                    color: AppColors.textSecondary,
                    fontSize: 14,
                  ),
                  textAlign: TextAlign.center,
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
    final totalOrders = allOrders.length;
    final activeOrders = allOrders
        .where((o) =>
            o.status == OrderStatus.pending ||
            o.status == OrderStatus.confirmed ||
            o.status == OrderStatus.preparing ||
            o.status == OrderStatus.outForDelivery)
        .length;
    final allLabOrders = ref.watch(userLabOrdersProvider).valueOrNull ?? [];
    final activeLabOrders = allLabOrders
        .where((o) =>
            o.status == LabOrderStatus.pending ||
            o.status == LabOrderStatus.sampleCollected ||
            o.status == LabOrderStatus.processing)
        .length;

    return CustomScrollView(
      slivers: [
        ProfileHeaderWidget(
          user: user,
          totalOrders: totalOrders,
          activeOrders: activeOrders,
          activeLabOrders: activeLabOrders,
          onEditTap: _showEditProfileSheet,
          onOrdersTap: () => context.go('/orders'),
          onLabTestsTap: () => context.push('/lab/orders'),
        ),
        if (isAdmin) ...[
          ProfileMenuSection(
            title: 'ADMIN',
            icon: Icons.admin_panel_settings_rounded,
            iconColor: AppColors.primary,
            children: [
              ProfileMenuTile(
                icon: Icons.receipt_long_rounded,
                iconColor: AppColors.primary,
                iconBgColor: const Color(0xFFE1F5EE),
                title: 'Admin Orders',
                subtitle: 'Manage customer orders',
                onTap: () => context.push('/admin'),
              ),
              ProfileMenuTile(
                icon: Icons.medication_rounded,
                iconColor: const Color(0xFF8E24AA),
                iconBgColor: const Color(0xFFF3E5F5),
                title: 'Prescriptions',
                subtitle: 'Review prescriptions',
                onTap: () => context.push('/admin/prescriptions'),
                isLast: true,
              ),
            ],
          ),
        ],
        ProfileMenuSection(
          title: 'ACCOUNT',
          icon: Icons.person_rounded,
          iconColor: AppColors.primary,
          children: [
            ProfileMenuTile(
              icon: Icons.receipt_long_rounded,
              iconColor: AppColors.primary,
              iconBgColor: const Color(0xFFE1F5EE),
              title: 'My Orders',
              subtitle: 'Track & manage your orders',
              trailing: activeOrders > 0
                  ? ProfileMenuBadge(text: '$activeOrders active')
                  : null,
              onTap: () => context.go('/orders'),
            ),
            ProfileMenuTile(
              icon: Icons.biotech_rounded,
              iconColor: AppColors.labPrimary,
              iconBgColor: AppColors.labPrimaryLight,
              title: 'My Lab Tests',
              subtitle: 'View & track lab bookings',
              trailing: activeLabOrders > 0
                  ? ProfileMenuBadge(
                      text: '$activeLabOrders active',
                      color: AppColors.labPrimary,
                    )
                  : null,
              onTap: () => context.push('/lab/orders'),
            ),
            ProfileMenuTile(
              icon: Icons.location_on_outlined,
              iconColor: const Color(0xFF1E88E5),
              iconBgColor: const Color(0xFFE3F2FD),
              title: 'Saved Addresses',
              subtitle: 'Home, Work & more',
              onTap: _showAddressSheet,
            ),
            ProfileMenuTile(
              icon: Icons.description_outlined,
              iconColor: const Color(0xFF8E24AA),
              iconBgColor: const Color(0xFFF3E5F5),
              title: 'My Prescriptions',
              subtitle: 'Medicine prescriptions',
              onTap: () => context.push('/my-prescriptions'),
            ),
            ProfileMenuTile(
              icon: Icons.assignment_outlined,
              iconColor: AppColors.labPrimary,
              iconBgColor: AppColors.labPrimaryLight,
              title: 'Lab Prescriptions',
              subtitle: 'Uploaded lab prescriptions',
              onTap: () => context.push('/my-prescriptions?type=lab'),
              isLast: true,
            ),
          ],
        ),
        ProfileMenuSection(
          title: 'SUPPORT',
          icon: Icons.headset_mic_rounded,
          iconColor: const Color(0xFFFB8C00),
          children: [
            ProfileMenuTile(
              icon: Icons.chat_bubble_outline,
              iconColor: const Color(0xFFFB8C00),
              iconBgColor: const Color(0xFFFFF3E0),
              title: 'Chat with Us',
              subtitle: 'WhatsApp support',
              onTap: () async {
                final message = Uri.encodeComponent(
                    'Hello Sushrut Aushadhi! I need help with my order.');
                final url =
                    'https://wa.me/${RemoteConfigService.storePhone}?text=$message';
                await _launchUrl(url);
              },
            ),
            ProfileMenuTile(
              icon: Icons.call_outlined,
              iconColor: const Color(0xFF1E88E5),
              iconBgColor: const Color(0xFFE3F2FD),
              title: 'Call Store',
              subtitle: '+91 80 XXXX XXXX',
              onTap: () => _launchUrl('tel:+918000000000'),
            ),
            ProfileMenuTile(
              icon: Icons.star_outline,
              iconColor: const Color(0xFFFFB300),
              iconBgColor: const Color(0xFFFFF8E1),
              title: 'Rate the App',
              subtitle: 'Share your feedback',
              onTap: () => _launchUrl('https://play.google.com/store/apps'),
              isLast: true,
            ),
          ],
        ),
        ProfileMenuSection(
          title: 'LEGAL',
          icon: Icons.gavel_rounded,
          iconColor: AppColors.textSecondary,
          children: [
            ProfileMenuTile(
              icon: Icons.info_outline_rounded,
              iconColor: AppColors.primary,
              iconBgColor: const Color(0xFFE1F5EE),
              title: 'About Sushrut Aushadhi',
              subtitle: 'Our story, license & contact',
              onTap: () => _showAboutDialog(context),
            ),
            ProfileMenuTile(
              icon: Icons.lock_outline,
              iconColor: AppColors.primary,
              iconBgColor: const Color(0xFFE1F5EE),
              title: 'Privacy Policy',
              onTap: () => _launchUrl('https://sushrutaushadhi.com/privacy'),
            ),
            ProfileMenuTile(
              icon: Icons.description_outlined,
              iconColor: AppColors.primary,
              iconBgColor: const Color(0xFFE1F5EE),
              title: 'Terms of Service',
              onTap: () => _launchUrl('https://sushrutaushadhi.com/terms'),
              isLast: true,
            ),
          ],
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
            child: Column(
              children: [
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _logout,
                    icon: const Icon(Icons.logout_rounded),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFFFEBEE),
                      foregroundColor: AppColors.error,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
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
                const SizedBox(height: 24),
                GestureDetector(
                  onTap: _testCrash,
                  child: Text(
                    'Sushrut Aushadhi v${AppStrings.appVersion}',
                    style: GoogleFonts.sora(
                      color: AppColors.textSecondary,
                      fontSize: 12,
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
  ConsumerState<EditProfileBottomSheet> createState() =>
      _EditProfileBottomSheetState();
}

class _EditProfileBottomSheetState
    extends ConsumerState<EditProfileBottomSheet> {
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
        SnackBar(
            content: Text('Profile updated successfully',
                style: GoogleFonts.sora())),
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
      padding:
          EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(
            24, 24, 24, 24 + MediaQuery.of(context).padding.bottom),
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
              Text('Edit Profile',
                  style: GoogleFonts.sora(
                      fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 20),
              TextFormField(
                controller: _nameController,
                decoration: InputDecoration(
                  labelText: 'Name',
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _addressController,
                decoration: InputDecoration(
                  labelText: 'Address',
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                maxLines: 3,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _pincodeController,
                decoration: InputDecoration(
                  labelText: 'Pincode',
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12)),
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
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
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
                          style: GoogleFonts.sora(
                              fontWeight: FontWeight.w600, fontSize: 16),
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
        SnackBar(
            content:
                Text('Address saved successfully', style: GoogleFonts.sora())),
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
      padding:
          EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(
            24, 24, 24, 24 + MediaQuery.of(context).padding.bottom),
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
              Text('Saved Addresses',
                  style: GoogleFonts.sora(
                      fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 20),
              TextFormField(
                controller: _phoneController,
                decoration: InputDecoration(
                  labelText: 'Phone Number',
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                keyboardType: TextInputType.phone,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _addressController,
                decoration: InputDecoration(
                  labelText: 'Address',
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                maxLines: 3,
                style: GoogleFonts.sora(),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _pincodeController,
                decoration: InputDecoration(
                  labelText: 'Pincode',
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12)),
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
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
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
                          style: GoogleFonts.sora(
                              fontWeight: FontWeight.w600, fontSize: 16),
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
