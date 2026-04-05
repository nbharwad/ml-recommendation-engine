import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:photo_view/photo_view.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../core/constants/app_colors.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../models/prescription_model.dart';
import '../../providers/auth_provider.dart';
import '../../services/prescription_service.dart';

class AdminPrescriptionsScreen extends ConsumerStatefulWidget {
  const AdminPrescriptionsScreen({super.key});

  @override
  ConsumerState<AdminPrescriptionsScreen> createState() =>
      _AdminPrescriptionsScreenState();
}

class _AdminPrescriptionsScreenState
    extends ConsumerState<AdminPrescriptionsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  PrescriptionType _selectedType = PrescriptionType.medicine;
  int _pendingCount = 0;
  int _approvedCount = 0;
  int _rejectedCount = 0;
  int _medicinePendingCount = 0;
  int _labPendingCount = 0;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _onRefresh() async {
    setState(() {});
    await Future.delayed(const Duration(milliseconds: 500));
  }

  @override
  Widget build(BuildContext context) {
    final isAdmin = ref.watch(isAdminProvider);

    if (!isAdmin) {
      return Scaffold(
        backgroundColor: AppColors.background,
        appBar: AppBar(
          backgroundColor: AppColors.primary,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back, color: Colors.white),
            onPressed: () => context.go('/home'),
          ),
          title: Text(
            'Access Denied',
            style: GoogleFonts.sora(color: Colors.white),
          ),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text('🔒', style: TextStyle(fontSize: 64)),
              const SizedBox(height: 16),
              Text(
                'Admin Access Required',
                style: GoogleFonts.sora(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'You do not have permission to view this page',
                style: GoogleFonts.sora(color: AppColors.textSecondary),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: AppColors.background,
      body: Column(
        children: [
          _buildCustomAppBar(),
          _buildTypeToggle(),
          _buildStatsRow(),
          _buildTabBar(),
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildPrescriptionList(null),
                _buildPrescriptionList(PrescriptionStatus.pending),
                _buildPrescriptionList(PrescriptionStatus.approved),
                _buildPrescriptionList(PrescriptionStatus.rejected),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCustomAppBar() {
    return Container(
      padding: EdgeInsets.only(
        top: MediaQuery.of(context).padding.top + 8,
        left: 16,
        right: 16,
        bottom: 12,
      ),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF0F6E56), Color(0xFF1D9E75)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Prescriptions',
                  style: GoogleFonts.sora(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Sushrut Aushadhi · Admin',
                  style: GoogleFonts.sora(
                    color: Colors.white.withOpacity(0.65),
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: () => setState(() {}),
          ),
        ],
      ),
    );
  }

  Widget _buildTypeToggle() {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 10),
      child: SegmentedButton<PrescriptionType>(
        segments: [
          ButtonSegment<PrescriptionType>(
            value: PrescriptionType.medicine,
            label: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('Medicine Rx'),
                if (_medicinePendingCount > 0) ...[
                  const SizedBox(width: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.orange,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '$_medicinePendingCount',
                      style: GoogleFonts.sora(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ],
            ),
            icon: const Icon(Icons.medication_outlined),
          ),
          ButtonSegment<PrescriptionType>(
            value: PrescriptionType.lab,
            label: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('Lab Rx'),
                if (_labPendingCount > 0) ...[
                  const SizedBox(width: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.orange,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '$_labPendingCount',
                      style: GoogleFonts.sora(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ],
            ),
            icon: const Icon(Icons.science_outlined),
          ),
        ],
        selected: {_selectedType},
        onSelectionChanged: (newSelection) {
          setState(() {
            _selectedType = newSelection.first;
            _tabController.index = 0;
          });
        },
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith<Color?>(
            (states) => states.contains(WidgetState.selected)
                ? AppColors.primary.withOpacity(0.12)
                : null,
          ),
          foregroundColor: WidgetStateProperty.resolveWith<Color?>(
            (states) => states.contains(WidgetState.selected)
                ? AppColors.primary
                : AppColors.textSecondary,
          ),
          side: WidgetStateProperty.all(
            const BorderSide(color: Color(0xFFDDDDDD)),
          ),
        ),
      ),
    );
  }

  Widget _buildStatsRow() {
    final screenWidth = MediaQuery.of(context).size.width;
    final isCompact = screenWidth < 360;
    
    return Container(
      padding: EdgeInsets.symmetric(horizontal: isCompact ? 8 : 12, vertical: isCompact ? 8 : 12),
      child: Row(
        children: [
          Expanded(child: _buildStatCard('Pending', _pendingCount.toString(), Colors.orange, isCompact)),
          SizedBox(width: isCompact ? 6 : 8),
          Expanded(child: _buildStatCard('Approved', _approvedCount.toString(), AppColors.primary, isCompact)),
          SizedBox(width: isCompact ? 6 : 8),
          Expanded(child: _buildStatCard('Rejected', _rejectedCount.toString(), AppColors.error, isCompact)),
        ],
      ),
    );
  }

  Widget _buildStatCard(String label, String value, Color color, bool isCompact) {
    return Expanded(
      child: Container(
        padding: EdgeInsets.symmetric(vertical: isCompact ? 8 : 12, horizontal: isCompact ? 6 : 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              value,
              style: GoogleFonts.sora(
                color: color,
                fontSize: isCompact ? 16 : 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: isCompact ? 2 : 4),
            Text(
              label,
              style: GoogleFonts.sora(
                color: AppColors.textSecondary,
                fontSize: isCompact ? 10 : 11,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTabBar() {
    return Container(
      color: Colors.white,
      child: TabBar(
        controller: _tabController,
        labelColor: AppColors.primary,
        unselectedLabelColor: AppColors.textSecondary,
        indicatorColor: AppColors.primary,
        indicatorWeight: 3,
        tabs: [
          Tab(text: 'All'),
          Tab(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('Pending'),
                if (_pendingCount > 0) ...[
                  const SizedBox(width: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.orange,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '$_pendingCount',
                      style: GoogleFonts.sora(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          Tab(text: 'Approved'),
          Tab(text: 'Rejected'),
        ],
      ),
    );
  }

  Widget _buildPrescriptionList(PrescriptionStatus? status) {
    // Always stream all prescriptions and filter client-side by type
    final stream = PrescriptionService().getAllPrescriptions();

    return RefreshIndicator(
      color: Colors.white,
      backgroundColor: const Color(0xFF0F6E56),
      strokeWidth: 2.5,
      onRefresh: _onRefresh,
      child: StreamBuilder<List<PrescriptionModel>>(
        stream: stream,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }

          final all = snapshot.data ?? [];
          _updateStats(all);

          // Filter by selected type first, then by status tab
          final byType = all.where((p) => p.prescriptionType == _selectedType).toList();
          final prescriptions = status == null
              ? byType
              : byType.where((p) => p.status == status).toList();

          if (prescriptions.isEmpty) {
            return _buildEmptyState(status);
          }

          return ListView.builder(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(12),
            itemCount: prescriptions.length,
            itemBuilder: (context, index) {
              return _PrescriptionCard(
                prescription: prescriptions[index],
                onStatusUpdate: () => setState(() {}),
              );
            },
          );
        },
      ),
    );
  }

  void _updateStats(List<PrescriptionModel> all) {
    int pending = 0;
    int approved = 0;
    int rejected = 0;
    int medicinePending = 0;
    int labPending = 0;

    for (final prescription in all) {
      // Per-type pending counts (for segment badges)
      if (prescription.status == PrescriptionStatus.pending) {
        if (prescription.prescriptionType == PrescriptionType.medicine) {
          medicinePending++;
        } else {
          labPending++;
        }
      }

      // Status counts for the currently selected type
      if (prescription.prescriptionType == _selectedType) {
        switch (prescription.status) {
          case PrescriptionStatus.pending:
            pending++;
            break;
          case PrescriptionStatus.approved:
            approved++;
            break;
          case PrescriptionStatus.rejected:
            rejected++;
            break;
        }
      }
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          _pendingCount = pending;
          _approvedCount = approved;
          _rejectedCount = rejected;
          _medicinePendingCount = medicinePending;
          _labPendingCount = labPending;
        });
      }
    });
  }

  Widget _buildEmptyState(PrescriptionStatus? status) {
    final emptyContent = SizedBox(
      height: MediaQuery.of(context).size.height * 0.6,
      child: switch (status) {
        null => const EmptyStateWidget(
            emoji: '📋',
            title: 'No Prescriptions',
            subtitle: 'There are no prescriptions yet.',
          ),
        PrescriptionStatus.pending => const EmptyStateWidget(
            emoji: '⏳',
            title: 'No Pending Prescriptions',
            subtitle: 'There are no prescriptions waiting for review.',
          ),
        PrescriptionStatus.approved => const EmptyStateWidget(
            emoji: '✅',
            title: 'No Approved Prescriptions',
            subtitle: 'There are no approved prescriptions.',
          ),
        PrescriptionStatus.rejected => const EmptyStateWidget(
            emoji: '❌',
            title: 'No Rejected Prescriptions',
            subtitle: 'There are no rejected prescriptions.',
          ),
      },
    );
    return RefreshIndicator(
      color: Colors.white,
      backgroundColor: const Color(0xFF0F6E56),
      strokeWidth: 2.5,
      onRefresh: _onRefresh,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        child: emptyContent,
      ),
    );
  }
}

class _PrescriptionCard extends StatelessWidget {
  final PrescriptionModel prescription;
  final VoidCallback onStatusUpdate;

  const _PrescriptionCard({
    required this.prescription,
    required this.onStatusUpdate,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFEEEEEE)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(),
          _buildCustomerRow(),
          _buildImage(context),
          if (prescription.status == PrescriptionStatus.pending) _buildActions(context),
          _buildFooter(),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: const Color(0xFFEEEEEE))),
      ),
      child: Row(
        children: [
          Text(
            'PR-${prescription.id.substring(0, 4).toUpperCase()}',
            style: GoogleFonts.sora(
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            _getTimeAgo(prescription.createdAt),
            style: GoogleFonts.sora(
              color: AppColors.textSecondary,
              fontSize: 11,
            ),
          ),
          const Spacer(),
          _buildStatusBadge(),
        ],
      ),
    );
  }

  Widget _buildStatusBadge() {
    Color bgColor;
    Color textColor;

    switch (prescription.status) {
      case PrescriptionStatus.pending:
        bgColor = Colors.orange.withOpacity(0.1);
        textColor = Colors.orange;
        break;
      case PrescriptionStatus.approved:
        bgColor = AppColors.primary.withOpacity(0.1);
        textColor = AppColors.primary;
        break;
      case PrescriptionStatus.rejected:
        bgColor = AppColors.error.withOpacity(0.1);
        textColor = AppColors.error;
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        prescription.status.displayName,
        style: GoogleFonts.sora(
          color: textColor,
          fontWeight: FontWeight.w600,
          fontSize: 11,
        ),
      ),
    );
  }

  Widget _buildCustomerRow() {
    final initials = prescription.userName != null && prescription.userName!.isNotEmpty
        ? prescription.userName!
            .split(' ')
            .map((e) => e.isNotEmpty ? e[0] : '')
            .take(2)
            .join()
            .toUpperCase()
        : 'CU';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: const Color(0xFFEEEEEE))),
      ),
      child: Row(
        children: [
          Container(
            width: 30,
            height: 30,
            decoration: BoxDecoration(
              color: AppColors.primary,
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                initials,
                style: GoogleFonts.sora(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  prescription.userName ?? 'Customer',
                  style: GoogleFonts.sora(
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                  ),
                ),
                if (prescription.userPhone != null)
                  Text(
                    prescription.userPhone!,
                    style: GoogleFonts.sora(
                      color: AppColors.textSecondary,
                      fontSize: 10,
                    ),
                  ),
              ],
            ),
          ),
          if (prescription.userPhone != null && prescription.userPhone!.isNotEmpty)
            GestureDetector(
              onTap: () => _launchUrl('tel:${prescription.userPhone}'),
              child: Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.call, color: Colors.white, size: 16),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildImage(BuildContext context) {
    return GestureDetector(
      onTap: () => _showPrescriptionViewer(context),
      child: Container(
        height: 150,
        width: double.infinity,
        child: CachedNetworkImage(
          imageUrl: prescription.imageUrl,
          fit: BoxFit.cover,
          placeholder: (context, url) => const Center(
            child: CircularProgressIndicator(),
          ),
          errorWidget: (context, url, error) => const Center(
            child: Icon(Icons.broken_image, size: 48),
          ),
        ),
      ),
    );
  }

  Widget _buildActions(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        children: [
          Expanded(
            child: OutlinedButton(
              onPressed: () => _updateStatus(PrescriptionStatus.rejected),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppColors.error,
                side: const BorderSide(color: AppColors.error),
              ),
              child: Text('Reject', style: GoogleFonts.sora(fontSize: 12)),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: ElevatedButton(
              onPressed: () => _updateStatus(PrescriptionStatus.approved),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
              ),
              child: Text('Approve', style: GoogleFonts.sora(fontSize: 12)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFooter() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        children: [
          Icon(Icons.calendar_today, size: 12, color: AppColors.textSecondary),
          const SizedBox(width: 4),
          Text(
            _formatDate(prescription.createdAt),
            style: GoogleFonts.sora(
              color: AppColors.textSecondary,
              fontSize: 11,
            ),
          ),
          if (prescription.notes != null && prescription.notes!.isNotEmpty) ...[
            const SizedBox(width: 12),
            Icon(Icons.note, size: 12, color: AppColors.textSecondary),
            const SizedBox(width: 4),
            Expanded(
              child: Text(
                prescription.notes!,
                style: GoogleFonts.sora(
                  color: AppColors.textSecondary,
                  fontSize: 11,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ],
      ),
    );
  }

  void _showPrescriptionViewer(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.black,
        insetPadding: EdgeInsets.zero,
        child: Stack(
          children: [
            PhotoView(
              imageProvider: CachedNetworkImageProvider(prescription.imageUrl),
              minScale: PhotoViewComputedScale.contained,
              maxScale: PhotoViewComputedScale.covered * 2,
              backgroundDecoration: const BoxDecoration(color: Colors.black),
            ),
            Positioned(
              top: MediaQuery.of(context).padding.top + 8,
              right: 8,
              child: IconButton(
                icon: const Icon(Icons.close, color: Colors.white, size: 28),
                onPressed: () => Navigator.of(context).pop(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _updateStatus(PrescriptionStatus status) async {
    try {
      await PrescriptionService().updatePrescriptionStatus(
        prescriptionId: prescription.id,
        status: status.name,
      );
      onStatusUpdate();
    } catch (e) {
      // Handle error
    }
  }

  String _getTimeAgo(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);

    if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else {
      return '${difference.inDays}d ago';
    }
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }
}