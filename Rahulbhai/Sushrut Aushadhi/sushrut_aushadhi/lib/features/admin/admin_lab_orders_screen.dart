import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_storage/firebase_storage.dart';

import '../../core/constants/app_colors.dart';
import '../../models/lab_order_model.dart';
import '../../providers/lab_providers.dart';

class AdminLabOrdersScreen extends ConsumerStatefulWidget {
  const AdminLabOrdersScreen({super.key});

  @override
  ConsumerState<AdminLabOrdersScreen> createState() => _AdminLabOrdersScreenState();
}

class _AdminLabOrdersScreenState extends ConsumerState<AdminLabOrdersScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  String _searchQuery = '';

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
    ref.invalidate(allLabOrdersProvider);
    await Future<void>.delayed(const Duration(milliseconds: 400));
  }

  @override
  Widget build(BuildContext context) {
    final ordersAsync = ref.watch(allLabOrdersProvider);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        bottom: false,
        child: Column(
          children: [
            _buildCustomAppBar(),
            const SizedBox(height: 8),
            ordersAsync.when(
              data: (orders) => _buildDailySummaryCard(orders),
              loading: () => const SizedBox.shrink(),
              error: (_, __) => const SizedBox.shrink(),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: _buildTabBar(),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildOrderList(ordersAsync, 'pending'),
                  _buildOrderList(ordersAsync, 'sampleCollected'),
                  _buildOrderList(ordersAsync, 'processing'),
                  _buildOrderList(ordersAsync, 'completed'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDailySummaryCard(List<LabOrderModel> orders) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final todayOrders = orders.where((o) => o.createdAt.isAfter(today)).toList();
    
    final pendingCount = todayOrders.where((o) => o.status == LabOrderStatus.pending).length;
    final completedCount = todayOrders.where((o) => o.status == LabOrderStatus.completed).length;
    final pendingPayments = todayOrders.where((o) => o.paymentStatus != 'paid').length;
    final todayRevenue = todayOrders.where((o) => o.paymentStatus == 'paid').fold(0.0, (sum, o) => sum + o.totalAmount);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0F6E56), Color(0xFF1D9E75)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "Today's Summary",
                style: GoogleFonts.sora(
                  color: Colors.white.withOpacity(0.9),
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                '${today.day}/${today.month}/${today.year}',
                style: GoogleFonts.sora(
                  color: Colors.white.withOpacity(0.7),
                  fontSize: 11,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _buildSummaryItem('Total Orders', '${todayOrders.length}', Icons.science),
              _buildSummaryItem('Pending', '$pendingCount', Icons.hourglass_empty),
              _buildSummaryItem('Completed', '$completedCount', Icons.check_circle),
              _buildSummaryItem('Revenue', '\u20B9${todayRevenue.toStringAsFixed(0)}', Icons.currency_rupee),
            ],
          ),
          if (pendingPayments > 0) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.warning_amber, color: Colors.amber, size: 14),
                  const SizedBox(width: 6),
                  Text(
                    '$pendingPayments pending payment${pendingPayments > 1 ? 's' : ''}',
                    style: GoogleFonts.sora(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSummaryItem(String label, String value, IconData icon) {
    return Expanded(
      child: Column(
        children: [
          Icon(icon, color: Colors.white.withOpacity(0.8), size: 18),
          const SizedBox(height: 4),
          Text(
            value,
            style: GoogleFonts.sora(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            label,
            style: GoogleFonts.sora(
              color: Colors.white.withOpacity(0.7),
              fontSize: 9,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCustomAppBar() {
    return Container(
      padding: EdgeInsets.only(
        top: MediaQuery.of(context).padding.top > 0 ? 8 : 16,
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
                  'Lab Orders',
                  style: GoogleFonts.sora(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Manage diagnostic test orders',
                  style: GoogleFonts.sora(
                    color: Colors.white.withOpacity(0.75),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Container(
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.14),
              borderRadius: BorderRadius.circular(12),
            ),
            child: IconButton(
              icon: const Icon(Icons.refresh_rounded, color: Colors.white),
              onPressed: _onRefresh,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTabBar() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
      ),
      child: TabBar(
        controller: _tabController,
        isScrollable: true,
        tabAlignment: TabAlignment.start,
        labelColor: AppColors.primary,
        unselectedLabelColor: AppColors.textSecondary,
        indicatorColor: AppColors.primary,
        indicatorWeight: 3,
        labelStyle: GoogleFonts.sora(fontWeight: FontWeight.w700),
        unselectedLabelStyle: GoogleFonts.sora(fontWeight: FontWeight.w600),
        tabs: const [
          Tab(text: 'Pending'),
          Tab(text: 'Sample Collected'),
          Tab(text: 'Processing'),
          Tab(text: 'Completed'),
        ],
      ),
    );
  }

  Widget _buildOrderList(AsyncValue<List<LabOrderModel>> ordersAsync, String status) {
    return ordersAsync.when(
      data: (orders) {
        final filteredOrders = orders.where((order) => order.status.name == status).toList();

        if (filteredOrders.isEmpty) {
          return _buildEmptyState(status);
        }

        return RefreshIndicator(
          color: Colors.white,
          backgroundColor: const Color(0xFF0F6E56),
          onRefresh: _onRefresh,
          child: ListView.builder(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
            itemCount: filteredOrders.length,
            itemBuilder: (context, index) {
              return _buildOrderCard(filteredOrders[index]);
            },
          ),
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (error, _) => Center(
        child: Text('Error: $error', style: GoogleFonts.sora()),
      ),
    );
  }

  Widget _buildEmptyState(String status) {
    final emptyMessages = {
      'pending': ('No Pending Orders', 'No lab orders waiting for sample collection.'),
      'sampleCollected': ('No Sample Collected', 'No orders awaiting processing.'),
      'processing': ('No Processing Orders', 'No orders currently being processed.'),
      'completed': ('No Completed Orders', 'No completed lab orders yet.'),
    };

    final message = emptyMessages[status] ?? ('No Orders', 'No orders found.');

    return RefreshIndicator(
      color: Colors.white,
      backgroundColor: const Color(0xFF0F6E56),
      onRefresh: _onRefresh,
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
        children: [
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.45,
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.science_outlined, size: 64, color: AppColors.textSecondary),
                  const SizedBox(height: 16),
                  Text(message.$1, style: GoogleFonts.sora(fontSize: 18, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  Text(message.$2, style: GoogleFonts.sora(color: AppColors.textSecondary)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOrderCard(LabOrderModel order) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE8ECE7)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildCardHeader(order),
          _buildCustomerSection(order),
          _buildTestsSection(order),
          _buildStatusUpdateRow(order),
          _buildCardFooter(order),
        ],
      ),
    );
  }

  Widget _buildCardHeader(LabOrderModel order) {
    final statusColor = _getStatusColor(order.status);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0xFFEFF3EF))),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              'SA-LB-${order.orderId.substring(0, 4).toUpperCase()}',
              style: GoogleFonts.sora(fontWeight: FontWeight.bold, fontSize: 14),
            ),
          ),
          Text(
            _getTimeAgo(order.createdAt),
            style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 11),
          ),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: statusColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(999),
            ),
            child: Text(
              order.status.displayName,
              style: GoogleFonts.sora(color: statusColor, fontWeight: FontWeight.w600, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCustomerSection(LabOrderModel order) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0xFFEFF3EF))),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 18,
            backgroundColor: const Color(0xFF00897B),
            child: Text(
              order.userName.isNotEmpty ? order.userName[0].toUpperCase() : 'U',
              style: GoogleFonts.sora(color: Colors.white, fontWeight: FontWeight.bold),
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
                const SizedBox(height: 2),
                Text(
                  order.userPhone,
                  style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 11),
                ),
                if (order.homeCollectionAddress != null && order.homeCollectionAddress!.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(
                    order.homeCollectionAddress!,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 11),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTestsSection(LabOrderModel order) {
    return Padding(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${order.testCount} test${order.testCount == 1 ? '' : 's'}',
            style: GoogleFonts.sora(fontSize: 11, color: AppColors.textSecondary, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          ...order.tests.take(3).map<Widget>((test) => Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              children: [
                Container(
                  width: 6,
                  height: 6,
                  decoration: const BoxDecoration(color: Color(0xFF00897B), shape: BoxShape.circle),
                ),
                const SizedBox(width: 8),
                Expanded(child: Text(test.testName, style: GoogleFonts.sora(fontSize: 12))),
                Text('\u20B9${test.price.toStringAsFixed(0)}', style: GoogleFonts.sora(fontSize: 12)),
              ],
            ),
          )),
          if (order.testCount > 3)
            Text(
              '+${order.testCount - 3} more',
              style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 11, fontStyle: FontStyle.italic),
            ),
        ],
      ),
    );
  }

  Widget _buildStatusUpdateRow(LabOrderModel order) {
    final nextStatuses = _getNextStatuses(order.status);
    if (nextStatuses.isEmpty) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'UPDATE STATUS',
            style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 1),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: nextStatuses.map((status) => InkWell(
              borderRadius: BorderRadius.circular(999),
              onTap: () => _updateOrderStatus(order.orderId, status),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  status.displayName,
                  style: GoogleFonts.sora(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
                ),
              ),
            )).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildCardFooter(LabOrderModel order) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Total Amount', style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 11)),
              Text('\u20B9${order.totalAmount.toStringAsFixed(0)}', style: GoogleFonts.sora(fontWeight: FontWeight.bold, fontSize: 16)),
              const SizedBox(height: 4),
              _buildPaymentStatusChip(order),
            ],
          ),
          Row(
            children: [
              if (order.paymentStatus != 'paid')
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: OutlinedButton(
                    onPressed: () => _markAsPaid(order.orderId),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppColors.primary,
                      side: const BorderSide(color: AppColors.primary),
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    ),
                    child: Text('Mark as Paid', style: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600)),
                  ),
                ),
              ElevatedButton(
                onPressed: () => context.push('/admin/lab-order/${order.orderId}'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                ),
                child: Text('View Details', style: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600)),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPaymentStatusChip(LabOrderModel order) {
    final isPaid = order.paymentStatus == 'paid';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: isPaid ? AppColors.primary.withOpacity(0.12) : Colors.orange.withOpacity(0.12),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        isPaid ? 'Paid' : 'Pending Payment',
        style: GoogleFonts.sora(
          color: isPaid ? AppColors.primary : Colors.orange,
          fontSize: 10,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Future<void> _markAsPaid(String orderId) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Confirm Payment', style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
        content: Text('Mark this order as paid?', style: GoogleFonts.sora()),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: Text('Cancel', style: GoogleFonts.sora(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('Confirm', style: GoogleFonts.sora(color: AppColors.primary)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      try {
        await ref.read(labServiceProvider).updatePaymentStatus(orderId, 'paid');
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Payment marked as paid', style: GoogleFonts.sora())),
        );
      } catch (e) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e', style: GoogleFonts.sora())),
        );
      }
    }
  }

  List<LabOrderStatus> _getNextStatuses(LabOrderStatus current) {
    switch (current) {
      case LabOrderStatus.pending:
        return [LabOrderStatus.sampleCollected, LabOrderStatus.cancelled];
      case LabOrderStatus.sampleCollected:
        return [LabOrderStatus.processing];
      case LabOrderStatus.processing:
        return [LabOrderStatus.completed];
      case LabOrderStatus.completed:
      case LabOrderStatus.cancelled:
        return [];
    }
  }

  Future<void> _updateOrderStatus(String orderId, LabOrderStatus newStatus) async {
    try {
      await ref.read(labServiceProvider).updateLabOrderStatus(orderId, newStatus);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Status updated to ${newStatus.displayName}', style: GoogleFonts.sora())),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e', style: GoogleFonts.sora())),
      );
    }
  }

  Color _getStatusColor(LabOrderStatus status) {
    switch (status) {
      case LabOrderStatus.pending:
        return Colors.orange;
      case LabOrderStatus.sampleCollected:
        return Colors.blue;
      case LabOrderStatus.processing:
        return Colors.purple;
      case LabOrderStatus.completed:
        return AppColors.primary;
      case LabOrderStatus.cancelled:
        return AppColors.error;
    }
  }

  String _getTimeAgo(DateTime dateTime) {
    final difference = DateTime.now().difference(dateTime);
    if (difference.inMinutes < 60) return '${difference.inMinutes}m ago';
    if (difference.inHours < 24) return '${difference.inHours}h ago';
    return '${difference.inDays}d ago';
  }
}

class AdminLabOrderDetailScreen extends ConsumerStatefulWidget {
  final String orderId;

  const AdminLabOrderDetailScreen({super.key, required this.orderId});

  @override
  ConsumerState<AdminLabOrderDetailScreen> createState() => _AdminLabOrderDetailScreenState();
}

class _AdminLabOrderDetailScreenState extends ConsumerState<AdminLabOrderDetailScreen> {
  final ValueNotifier<double?> _uploadProgress = ValueNotifier<double?>(null);

  @override
  void dispose() {
    _uploadProgress.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final orderAsync = ref.watch(labOrderProvider(widget.orderId));

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
        title: Text('Lab Order Details', style: GoogleFonts.sora()),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.invalidate(labOrderProvider(widget.orderId)),
          ),
        ],
      ),
      body: orderAsync.when(
        data: (order) {
          if (order == null) {
            return const Center(child: Text('Order not found'));
          }
          return _buildContent(context, ref, order);
        },
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(child: Text('Error: $error')),
      ),
    );
  }

  Widget _buildContent(BuildContext context, WidgetRef ref, LabOrderModel order) {
    final statusColor = _getStatusColor(order.status);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFE8ECE7)),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: statusColor.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(_getStatusIcon(order.status), color: statusColor, size: 28),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('SA-LB-${order.orderId.substring(0, 4).toUpperCase()}', style: GoogleFonts.sora(fontSize: 16, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: statusColor.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Text(order.status.displayName, style: GoogleFonts.sora(color: statusColor, fontSize: 12, fontWeight: FontWeight.w600)),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFE8ECE7)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Customer', style: GoogleFonts.sora(fontSize: 14, fontWeight: FontWeight.bold, color: AppColors.textSecondary)),
                const SizedBox(height: 12),
                Row(
                  children: [
                    CircleAvatar(backgroundColor: const Color(0xFF00897B), child: Text(order.userName.isNotEmpty ? order.userName[0].toUpperCase() : 'U', style: const TextStyle(color: Colors.white))),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(order.userName.isNotEmpty ? order.userName : 'Customer', style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
                          Text(order.userPhone, style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 12)),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFE8ECE7)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Tests (${order.testCount})', style: GoogleFonts.sora(fontSize: 14, fontWeight: FontWeight.bold, color: AppColors.textSecondary)),
                const SizedBox(height: 12),
                ...order.tests.map<Widget>((test) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(
                    children: [
                      const Icon(Icons.science, size: 16, color: Color(0xFF00897B)),
                      const SizedBox(width: 8),
                      Expanded(child: Text(test.testName, style: GoogleFonts.sora(fontSize: 14))),
                      Text('\u20B9${test.price.toStringAsFixed(0)}', style: GoogleFonts.sora(fontWeight: FontWeight.w600)),
                    ],
                  ),
                )),
                const Divider(),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('Total', style: GoogleFonts.sora(fontSize: 16, fontWeight: FontWeight.bold)),
                    Text('\u20B9${order.totalAmount.toStringAsFixed(0)}', style: GoogleFonts.sora(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.primary)),
                  ],
                ),
              ],
            ),
          ),
          if (order.homeCollectionAddress != null && order.homeCollectionAddress!.isNotEmpty) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFE8ECE7)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.location_on, color: Color(0xFF1E88E5), size: 20),
                      const SizedBox(width: 8),
                      Text('Sample Collection Address', style: GoogleFonts.sora(fontSize: 14, fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(order.homeCollectionAddress!, style: GoogleFonts.sora(fontSize: 14, color: AppColors.textSecondary)),
                ],
              ),
            ),
          ],
          if (order.notes != null && order.notes!.isNotEmpty) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFE8ECE7)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.note, color: Color(0xFFFB8C00), size: 20),
                      const SizedBox(width: 8),
                      Text('Notes', style: GoogleFonts.sora(fontSize: 14, fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(order.notes!, style: GoogleFonts.sora(fontSize: 14, color: AppColors.textSecondary)),
                ],
              ),
            ),
          ],
          const SizedBox(height: 24),
          Text('Upload Lab Result', style: GoogleFonts.sora(fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          _buildUploadSection(context, ref, order),
          const SizedBox(height: 24),
          Text('Update Status', style: GoogleFonts.sora(fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _getNextStatuses(order.status)
                .map((status) => ElevatedButton(
                  onPressed: () => _updateStatus(context, ref, order.orderId, status),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: status == LabOrderStatus.cancelled ? AppColors.error : AppColors.primary,
                    foregroundColor: Colors.white,
                  ),
                  child: Text(status.displayName, style: GoogleFonts.sora()),
                ))
                .toList(),
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  List<LabOrderStatus> _getNextStatuses(LabOrderStatus current) {
    switch (current) {
      case LabOrderStatus.pending:
        return [LabOrderStatus.sampleCollected, LabOrderStatus.cancelled];
      case LabOrderStatus.sampleCollected:
        return [LabOrderStatus.processing];
      case LabOrderStatus.processing:
        return [LabOrderStatus.completed];
      case LabOrderStatus.completed:
      case LabOrderStatus.cancelled:
        return [];
    }
  }

  Future<void> _updateStatus(BuildContext context, WidgetRef ref, String orderId, LabOrderStatus status) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Confirm Status', style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
        content: Text('Update order status to ${status.displayName}?', style: GoogleFonts.sora()),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: Text('Cancel', style: GoogleFonts.sora(color: AppColors.textSecondary))),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('Confirm', style: GoogleFonts.sora(color: AppColors.primary)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      try {
        await ref.read(labServiceProvider).updateLabOrderStatus(orderId, status);
        if (!context.mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Status updated to ${status.displayName}', style: GoogleFonts.sora())),
        );
      } catch (e) {
        if (!context.mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e', style: GoogleFonts.sora())),
        );
      }
    }
  }

  Color _getStatusColor(LabOrderStatus status) {
    switch (status) {
      case LabOrderStatus.pending:
        return Colors.orange;
      case LabOrderStatus.sampleCollected:
        return Colors.blue;
      case LabOrderStatus.processing:
        return Colors.purple;
      case LabOrderStatus.completed:
        return AppColors.primary;
      case LabOrderStatus.cancelled:
        return AppColors.error;
    }
  }

  IconData _getStatusIcon(LabOrderStatus status) {
    switch (status) {
      case LabOrderStatus.pending:
        return Icons.hourglass_empty;
      case LabOrderStatus.sampleCollected:
        return Icons.bloodtype;
      case LabOrderStatus.processing:
        return Icons.science;
      case LabOrderStatus.completed:
        return Icons.check_circle;
      case LabOrderStatus.cancelled:
        return Icons.cancel;
    }
  }

  Widget _buildUploadSection(BuildContext context, WidgetRef ref, LabOrderModel order) {
    return ValueListenableBuilder<double?>(
      builder: (context, progress, child) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (order.labResultUrl != null && order.labResultUrl!.isNotEmpty) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.primary.withOpacity(0.3)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.check_circle, color: AppColors.primary, size: 24),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Lab Result Uploaded', style: GoogleFonts.sora(fontWeight: FontWeight.w600)),
                          Text('PDF available', style: GoogleFonts.sora(color: AppColors.textSecondary, fontSize: 12)),
                        ],
                      ),
                    ),
                    TextButton(
                      onPressed: () => _updateStatus(context, ref, order.orderId, LabOrderStatus.completed),
                      child: Text('Mark Complete', style: GoogleFonts.sora(fontSize: 12)),
                    ),
                  ],
                ),
              ),
            ] else if (progress != null) ...[
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.upload_file, color: AppColors.primary),
                      const SizedBox(width: 8),
                      Text('Uploading... ${(progress * 100).toInt()}%', style: GoogleFonts.sora()),
                    ],
                  ),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(value: progress, backgroundColor: Colors.grey[300], valueColor: const AlwaysStoppedAnimation<Color>(AppColors.primary)),
                ],
              ),
            ] else ...[
              InkWell(
                onTap: () => _pickAndUploadFile(context, ref, order.orderId),
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFE8ECE7), style: BorderStyle.solid),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.cloud_upload_outlined, color: AppColors.primary),
                      const SizedBox(width: 8),
                      Text('Upload PDF Report', style: GoogleFonts.sora(color: AppColors.primary, fontWeight: FontWeight.w600)),
                    ],
                  ),
                ),
              ),
            ],
          ],
        );
      },
      valueListenable: _uploadProgress,
    );
  }

  Future<void> _pickAndUploadFile(BuildContext context, WidgetRef ref, String orderId) async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        allowMultiple: false,
      );

      if (result == null || result.files.isEmpty) return;

      final file = result.files.first;
      if (file.path == null) {
        if (!context.mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not access file', style: GoogleFonts.sora())),
        );
        return;
      }

      _uploadProgress.value = 0.0;

      final uploadTask = FirebaseStorage.instance.ref().child('lab_results').child(orderId).child(file.name).putFile(
        File(file.path!),
        SettableMetadata(contentType: 'application/pdf'),
      );

      uploadTask.snapshotEvents.listen((task) {
        if (task.state == TaskState.running) {
          _uploadProgress.value = task.bytesTransferred / task.totalBytes;
        } else if (task.state == TaskState.success) {
          _uploadProgress.value = null;
        }
      });

      final snapshot = await uploadTask;
      final downloadUrl = await snapshot.ref.getDownloadURL();

      await ref.read(labServiceProvider).updateLabOrderStatus(orderId, LabOrderStatus.completed, note: 'Lab result uploaded');

      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Lab result uploaded successfully', style: GoogleFonts.sora())),
      );
    } catch (e) {
      _uploadProgress.value = null;
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error uploading: $e', style: GoogleFonts.sora())),
      );
    }
  }
}