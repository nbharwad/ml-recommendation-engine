import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../core/constants/app_colors.dart';
import '../../models/lab_order_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/lab_providers.dart';
import 'admin_orders_screen.dart';

class AdminShellScreen extends ConsumerStatefulWidget {
  const AdminShellScreen({super.key});

  @override
  ConsumerState<AdminShellScreen> createState() => _AdminShellScreenState();
}

class _AdminShellScreenState extends ConsumerState<AdminShellScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
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
          title: Text('Access Denied', style: GoogleFonts.sora(color: Colors.white)),
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.lock_outline, size: 64, color: AppColors.primary),
                const SizedBox(height: 16),
                Text('Admin Access Required',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.sora(fontSize: 20, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Text('You do not have permission to view this page.',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.sora(color: AppColors.textSecondary)),
              ],
            ),
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        bottom: false,
        child: Column(
          children: [
            _buildAppBar(),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                physics: const NeverScrollableScrollPhysics(),
                children: const [
                  AdminOrdersScreen(),
                  _LabDashboard(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    return Container(
      padding: EdgeInsets.only(
        top: MediaQuery.of(context).padding.top > 0 ? 8 : 16,
        left: 16,
        right: 16,
        bottom: 0,
      ),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF0F6E56), Color(0xFF1D9E75)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              IconButton(
                onPressed: () => context.go('/profile'),
                icon: const Icon(Icons.arrow_back, color: Colors.white),
              ),
              const SizedBox(width: 4),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Admin Panel',
                      style: GoogleFonts.sora(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Sushrut Aushadhi · Operations',
                      style: GoogleFonts.sora(
                        color: Colors.white.withValues(alpha: 0.75),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          TabBar(
            controller: _tabController,
            indicatorColor: Colors.white,
            indicatorWeight: 3,
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white.withValues(alpha: 0.55),
            labelStyle: GoogleFonts.sora(fontWeight: FontWeight.bold, fontSize: 14),
            unselectedLabelStyle: GoogleFonts.sora(fontWeight: FontWeight.w500, fontSize: 14),
            tabs: const [
              Tab(icon: Icon(Icons.medication_outlined), text: 'Medicine'),
              Tab(icon: Icon(Icons.science_outlined), text: 'Lab'),
            ],
          ),
        ],
      ),
    );
  }
}

// ── LAB DASHBOARD TAB ─────────────────────────────────────────────────────────

class _LabDashboard extends ConsumerWidget {
  const _LabDashboard();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ordersAsync = ref.watch(allLabOrdersProvider);

    return RefreshIndicator(
      color: Colors.white,
      backgroundColor: const Color(0xFF0F6E56),
      onRefresh: () async {
        // Stream is already real-time, no need to invalidate
        await Future<void>.delayed(const Duration(milliseconds: 400));
      },
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
        children: [
          // Stats
          ordersAsync.when(
            data: (orders) => _LabStatsGrid(orders: orders),
            loading: () => const SizedBox(
              height: 100,
              child: Center(child: CircularProgressIndicator()),
            ),
            error: (_, __) => const SizedBox.shrink(),
          ),
          const SizedBox(height: 20),

          // Section nav chips
          Text(
            'MANAGE',
            style: GoogleFonts.sora(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              color: AppColors.textSecondary,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 10),
          _LabNavGrid(),
          const SizedBox(height: 24),

          // Pending lab orders preview
          Row(
            children: [
              Text(
                'PENDING ORDERS',
                style: GoogleFonts.sora(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textSecondary,
                  letterSpacing: 1.2,
                ),
              ),
              const Spacer(),
              GestureDetector(
                onTap: () => context.push('/admin/lab-orders'),
                child: Text(
                  'See All →',
                  style: GoogleFonts.sora(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppColors.primary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ordersAsync.when(
            data: (orders) {
              final pending = orders
                  .where((o) => o.status == LabOrderStatus.pending)
                  .take(5)
                  .toList();
              if (pending.isEmpty) {
                return Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFE8ECE7)),
                  ),
                  child: Center(
                    child: Text(
                      'No pending lab orders',
                      style: GoogleFonts.sora(color: AppColors.textSecondary),
                    ),
                  ),
                );
              }
              return Column(
                children: pending
                    .map((order) => _PendingLabOrderCard(order: order))
                    .toList(),
              );
            },
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (e, _) => Text('Error: $e', style: GoogleFonts.sora()),
          ),
        ],
      ),
    );
  }
}

class _LabStatsGrid extends StatelessWidget {
  final List<LabOrderModel> orders;

  const _LabStatsGrid({required this.orders});

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final todayOrders = orders.where((o) => o.createdAt.isAfter(today)).toList();

    final pending = orders.where((o) => o.status == LabOrderStatus.pending).length;
    final completedToday = todayOrders.where((o) => o.status == LabOrderStatus.completed).length;
    final todayRevenue = todayOrders
        .where((o) => o.paymentStatus == 'paid')
        .fold(0.0, (sum, o) => sum + o.totalAmount);
    final pendingPayments = orders
        .where((o) => o.paymentStatus != 'paid' && o.status != LabOrderStatus.cancelled)
        .length;

    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 2.2,
      children: [
        _StatCard(label: 'Pending', value: '$pending', color: Colors.orange, icon: Icons.pending_actions_rounded),
        _StatCard(label: 'Completed Today', value: '$completedToday', color: AppColors.primary, icon: Icons.check_circle_outline_rounded),
        _StatCard(label: 'Today Revenue', value: 'Rs ${todayRevenue.toStringAsFixed(0)}', color: const Color(0xFF0E8E62), icon: Icons.payments_rounded),
        _StatCard(label: 'Pending Payments', value: '$pendingPayments', color: AppColors.error, icon: Icons.warning_amber_rounded),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  final IconData icon;

  const _StatCard({
    required this.label,
    required this.value,
    required this.color,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE8ECE7)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.sora(
                    color: color,
                    fontSize: 15,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  label,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.sora(
                    color: AppColors.textSecondary,
                    fontSize: 10,
                    height: 1.3,
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

class _LabNavGrid extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final items = [
      _LabNavItem('Lab Orders', Icons.science_outlined, const Color(0xFF0277BD), '/admin/lab-orders'),
      _LabNavItem('Lab Packages', Icons.inventory_2_outlined, const Color(0xFF6A1B9A), '/admin/lab-packages'),
      _LabNavItem('Manage Tests', Icons.biotech_outlined, const Color(0xFF00838F), '/admin/lab-tests'),
      _LabNavItem('Lab Prescriptions', Icons.description_outlined, const Color(0xFFE65100), '/admin/prescriptions'),
    ];

    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 10,
      mainAxisSpacing: 10,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 2.8,
      children: items.map((item) => _LabNavTile(item: item)).toList(),
    );
  }
}

class _LabNavItem {
  final String label;
  final IconData icon;
  final Color color;
  final String route;

  const _LabNavItem(this.label, this.icon, this.color, this.route);
}

class _LabNavTile extends StatelessWidget {
  final _LabNavItem item;

  const _LabNavTile({required this.item});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => context.push(item.route),
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: item.color.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: item.color.withValues(alpha: 0.25)),
        ),
        child: Row(
          children: [
            Icon(item.icon, color: item.color, size: 18),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                item.label,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.sora(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: item.color,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PendingLabOrderCard extends StatelessWidget {
  final LabOrderModel order;

  const _PendingLabOrderCard({required this.order});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => context.push('/admin/lab-order/${order.orderId}'),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFFE8ECE7)),
        ),
        child: Row(
          children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: const Color(0xFF00897B),
              child: Text(
                order.userName.isNotEmpty ? order.userName[0].toUpperCase() : 'U',
                style: GoogleFonts.sora(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    order.userName.isNotEmpty ? order.userName : 'Customer',
                    style: GoogleFonts.sora(fontWeight: FontWeight.bold, fontSize: 13),
                  ),
                  Text(
                    '${order.testCount} test${order.testCount == 1 ? '' : 's'} · Rs ${order.totalAmount.toStringAsFixed(0)}',
                    style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 11),
                  ),
                ],
              ),
            ),
            const Icon(Icons.arrow_forward_ios, size: 14, color: AppColors.textSecondary),
          ],
        ),
      ),
    );
  }
}
