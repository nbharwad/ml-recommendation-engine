import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../../core/constants/app_colors.dart';
import '../../core/utils/helpers.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../models/order_model.dart';
import '../../providers/cart_provider.dart';
import '../../providers/orders_provider.dart';

class OrdersScreen extends ConsumerStatefulWidget {
  const OrdersScreen({super.key});

  @override
  ConsumerState<OrdersScreen> createState() => _OrdersScreenState();
}

class _OrdersScreenState extends ConsumerState<OrdersScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  bool _isReordering = false;

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
    ref.invalidate(ordersProvider);
    await Future.delayed(const Duration(milliseconds: 500));
  }

  Future<void> _handleReorder(OrderModel order) async {
    if (_isReordering) return;

    setState(() => _isReordering = true);

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(
        child: Card(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(color: Color(0xFF0F6E56)),
                SizedBox(height: 16),
                Text('Adding items to cart...'),
              ],
            ),
          ),
        ),
      ),
    );

    try {
      await ref.read(cartProvider.notifier).reorderFromOrder(order);
      
      if (!mounted) return;
      Navigator.of(context).pop();

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Items added to cart'),
          backgroundColor: const Color(0xFF0F6E56),
          behavior: SnackBarBehavior.floating,
          action: SnackBarAction(
            label: 'View Cart',
            textColor: Colors.white,
            onPressed: () => context.push('/cart'),
          ),
        ),
      );
      context.push('/cart');
    } catch (e) {
      if (!mounted) return;
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isReordering = false);
      }
    }
  }

  void _showRatingDialog(OrderModel order) {
    int selectedRating = order.rating ?? 0;
    
    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: Text('Rate Order', style: GoogleFonts.sora(fontWeight: FontWeight.w600)),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'How was your experience?',
                style: GoogleFonts.sora(color: AppColors.textSecondary),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(5, (index) {
                  final starIndex = index + 1;
                  return GestureDetector(
                    onTap: () => setDialogState(() => selectedRating = starIndex),
                    child: Icon(
                      starIndex <= selectedRating ? Icons.star_rounded : Icons.star_outline_rounded,
                      color: const Color(0xFF0F6E56),
                      size: 36,
                    ),
                  );
                }),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: selectedRating == 0
                  ? null
                  : () async {
                      try {
                        await FirebaseFirestore.instance
                            .collection('orders')
                            .doc(order.orderId)
                            .update({'rating': selectedRating});
                        if (!mounted) return;
                        Navigator.pop(context);
                        ref.invalidate(ordersProvider);
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Thank you for your rating!'),
                            backgroundColor: Color(0xFF0F6E56),
                          ),
                        );
                      } catch (e) {
                        if (!mounted) return;
                        Navigator.pop(context);
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('Error: $e'),
                            backgroundColor: Colors.red,
                          ),
                        );
                      }
                    },
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF0F6E56),
                foregroundColor: Colors.white,
              ),
              child: const Text('Submit'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final ordersAsync = ref.watch(ordersProvider);
    final user = FirebaseAuth.instance.currentUser;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text(
          'My Orders',
          style: GoogleFonts.sora(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary,
          ),
        ),
        centerTitle: true,
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true,
          tabAlignment: TabAlignment.start,
          labelColor: AppColors.primary,
          unselectedLabelColor: AppColors.textSecondary,
          indicatorColor: AppColors.primary,
          indicatorWeight: 3,
          labelStyle: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600),
          unselectedLabelStyle: GoogleFonts.sora(fontSize: 12),
          tabs: const [
            Tab(text: 'All'),
            Tab(text: 'Active'),
            Tab(text: 'Delivered'),
            Tab(text: 'Cancelled'),
          ],
        ),
      ),
      body: ordersAsync.when(
        data: (orders) {
          if (user == null) {
            return _buildLoginPrompt();
          }
          if (orders.isEmpty) {
            return _buildEmptyState();
          }
          return TabBarView(
            controller: _tabController,
            children: [
              _buildOrdersList(orders, 'All'),
              _buildOrdersList(_filterOrders(orders, 'active'), 'Active'),
              _buildOrdersList(_filterOrders(orders, 'delivered'), 'Delivered'),
              _buildOrdersList(_filterOrders(orders, 'cancelled'), 'Cancelled'),
            ],
          );
        },
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(child: Text('Error: $error')),
      ),
    );
  }

  Widget _buildLoginPrompt() {
    return EmptyStateWidget(
      emoji: '🔐',
      title: 'Login to View Your Orders',
      subtitle: 'Sign in to see your order history and track your deliveries.',
      buttonText: 'Login',
      onButtonPressed: () => context.push('/login'),
    );
  }

  Widget _buildEmptyState() {
    return EmptyStateWidget(
      emoji: '📦',
      title: 'No Orders Yet',
      subtitle: 'You haven\'t placed any orders yet. Start shopping to see your orders here.',
      buttonText: 'Browse Medicines',
      onButtonPressed: () => context.go('/home'),
    );
  }

  List<OrderModel> _filterOrders(List<OrderModel> orders, String filter) {
    switch (filter) {
      case 'active':
        return orders.where((o) {
          return o.status == OrderStatus.pending ||
              o.status == OrderStatus.confirmed ||
              o.status == OrderStatus.preparing ||
              o.status == OrderStatus.outForDelivery;
        }).toList();
      case 'delivered':
        return orders.where((o) => o.status == OrderStatus.delivered).toList();
      case 'cancelled':
        return orders.where((o) => o.status == OrderStatus.cancelled).toList();
      default:
        return orders;
    }
  }

  Widget _buildOrdersList(List<OrderModel> orders, String filter) {
    if (orders.isEmpty) {
      return _buildTabEmptyState(filter);
    }
    return RefreshIndicator(
      color: const Color(0xFF0F6E56),
      backgroundColor: Colors.white,
      strokeWidth: 2.5,
      onRefresh: _onRefresh,
      child: ListView.builder(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(12),
        itemCount: orders.length,
        itemBuilder: (context, index) => _buildOrderCard(orders[index]),
      ),
    );
  }

  Widget _buildTabEmptyState(String filter) {
    final messages = {
      'All': RefreshIndicator(
        color: const Color(0xFF0F6E56),
        backgroundColor: Colors.white,
        strokeWidth: 2.5,
        onRefresh: _onRefresh,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: SizedBox(
            height: MediaQuery.of(context).size.height * 0.6,
            child: const EmptyStateWidget(
              emoji: '📋',
              title: 'No Orders Yet',
              subtitle: "You haven't placed any orders yet.",
            ),
          ),
        ),
      ),
      'Active': RefreshIndicator(
        color: const Color(0xFF0F6E56),
        backgroundColor: Colors.white,
        strokeWidth: 2.5,
        onRefresh: _onRefresh,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: SizedBox(
            height: MediaQuery.of(context).size.height * 0.6,
            child: const EmptyStateWidget(
              emoji: '🚚',
              title: 'No Active Orders',
              subtitle: "You don't have any orders in progress.",
            ),
          ),
        ),
      ),
      'Delivered': RefreshIndicator(
        color: const Color(0xFF0F6E56),
        backgroundColor: Colors.white,
        strokeWidth: 2.5,
        onRefresh: _onRefresh,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: SizedBox(
            height: MediaQuery.of(context).size.height * 0.6,
            child: const EmptyStateWidget(
              emoji: '✅',
              title: 'No Delivered Orders',
              subtitle: "You don't have any delivered orders yet.",
            ),
          ),
        ),
      ),
      'Cancelled': RefreshIndicator(
        color: const Color(0xFF0F6E56),
        backgroundColor: Colors.white,
        strokeWidth: 2.5,
        onRefresh: _onRefresh,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: SizedBox(
            height: MediaQuery.of(context).size.height * 0.6,
            child: const EmptyStateWidget(
              emoji: '❌',
              title: 'No Cancelled Orders',
              subtitle: "You don't have any cancelled orders.",
            ),
          ),
        ),
      ),
    };
    return messages[filter] ?? _buildEmptyState();
  }

  Widget _buildOrderCard(OrderModel order) {
    final isActive =
        order.status != OrderStatus.delivered && order.status != OrderStatus.cancelled;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        children: [
          _buildCardHeader(order),
          if (isActive) _buildStatusTimeline(order.status),
          _buildItemsPreview(order),
          _buildCardFooter(order),
        ],
      ),
    );
  }

  Widget _buildCardHeader(OrderModel order) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'SA-${order.orderId.substring(order.orderId.length - 4).toUpperCase()}',
                  style: GoogleFonts.sora(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  Helpers.formatDateTime(order.createdAt),
                  style: GoogleFonts.sora(fontSize: 11, color: AppColors.textSecondary),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Flexible(child: _buildStatusChip(order.status)),
        ],
      ),
    );
  }

  Widget _buildStatusChip(OrderStatus status) {
    late Color bgColor;
    late String label;
    IconData? icon;

    switch (status) {
      case OrderStatus.pending:
        bgColor = AppColors.statusPending;
        label = 'Pending';
        icon = Icons.hourglass_top_rounded;
        break;
      case OrderStatus.confirmed:
        bgColor = AppColors.statusConfirmed;
        label = 'Confirmed';
        icon = Icons.check_circle_outline;
        break;
      case OrderStatus.preparing:
        bgColor = AppColors.statusPreparing;
        label = 'Preparing';
        icon = Icons.inventory_2_outlined;
        break;
      case OrderStatus.outForDelivery:
        bgColor = AppColors.statusOutForDelivery;
        label = 'Out for Delivery';
        icon = Icons.local_shipping_outlined;
        break;
      case OrderStatus.delivered:
        bgColor = AppColors.statusDelivered;
        label = 'Delivered';
        icon = Icons.task_alt;
        break;
      case OrderStatus.cancelled:
        bgColor = AppColors.statusCancelled;
        label = 'Cancelled';
        icon = Icons.cancel_outlined;
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: bgColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 4,
        children: [
          if (icon != null) Icon(icon, size: 12, color: bgColor),
          Text(
            label,
            style: GoogleFonts.sora(fontSize: 11, fontWeight: FontWeight.w600, color: bgColor),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusTimeline(OrderStatus status) {
    final statuses = [
      OrderStatus.pending,
      OrderStatus.confirmed,
      OrderStatus.outForDelivery,
      OrderStatus.delivered,
    ];
    final currentIndex = statuses.indexOf(status);
    final displayStatus = status == OrderStatus.preparing ? 1 : currentIndex;
    final labels = ['Placed', 'Confirmed', 'On the way', 'Delivered'];

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: labels.asMap().entries.map((entry) {
          final index = entry.key;
          final label = entry.value;
          final isCompleted = index <= displayStatus;
          final isCurrent = index == displayStatus;

          return SizedBox(
            width: 92,
            child: Row(
              children: [
                Column(
                  children: [
                    Container(
                      width: 12,
                      height: 12,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isCompleted ? AppColors.primary : Colors.grey.shade300,
                        boxShadow: isCurrent
                            ? [
                                BoxShadow(
                                  color: AppColors.primary.withOpacity(0.5),
                                  blurRadius: 8,
                                  spreadRadius: 2,
                                ),
                              ]
                            : null,
                      ),
                    ),
                    const SizedBox(height: 4),
                    SizedBox(
                      width: 56,
                      child: Text(
                        label,
                        textAlign: TextAlign.center,
                        style: GoogleFonts.sora(
                          fontSize: 8,
                          color: isCurrent ? AppColors.primary : AppColors.textSecondary,
                          fontWeight: isCurrent ? FontWeight.w600 : FontWeight.normal,
                        ),
                      ),
                    ),
                  ],
                ),
                if (index < labels.length - 1)
                  Expanded(
                    child: Container(
                      height: 2,
                      margin: const EdgeInsets.only(bottom: 20),
                      color: index < displayStatus ? AppColors.primary : Colors.grey.shade300,
                    ),
                  ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildItemsPreview(OrderModel order) {
    final displayItems = order.items.take(2).toList();
    final remaining = order.items.length - displayItems.length;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        children: [
          ...displayItems.map((item) {
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: AppColors.primaryLight,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.medication_rounded, color: AppColors.primary, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          item.medicineName,
                          style: GoogleFonts.sora(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: AppColors.textPrimary,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        Text(
                          'Qty: ${item.quantity}',
                          style: GoogleFonts.sora(fontSize: 10, color: AppColors.textSecondary),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    '\u20B9${item.subtotal.toStringAsFixed(0)}',
                    style: GoogleFonts.sora(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: AppColors.textPrimary,
                    ),
                  ),
                ],
              ),
            );
          }),
          if (remaining > 0)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '+$remaining more',
                  style: GoogleFonts.sora(
                    fontSize: 12,
                    color: AppColors.primary,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildCardFooter(OrderModel order) {
    final isDelivered = order.status == OrderStatus.delivered;
    final isActive =
        order.status != OrderStatus.delivered && order.status != OrderStatus.cancelled;
    final isPending = order.status == OrderStatus.pending;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: Colors.grey.shade100)),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 360;

          final actions = isDelivered
              ? Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  alignment: WrapAlignment.end,
                  children: [
                    OutlinedButton(
                      onPressed: _isReordering ? null : () => _handleReorder(order),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppColors.primary,
                        side: const BorderSide(color: AppColors.primary),
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      child: Text('Reorder', style: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600)),
                    ),
                    ElevatedButton(
                      onPressed: order.rating != null ? null : () => _showRatingDialog(order),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                        elevation: 0,
                      ),
                      child: Text(order.rating != null ? 'Rated' : 'Rate', style: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600)),
                    ),
                  ],
                )
              : isActive && !isPending
                  ? OutlinedButton(
                      onPressed: () {},
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppColors.primary,
                        side: const BorderSide(color: AppColors.primary),
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      child: Text('Call Us', style: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600)),
                    )
                  : isPending
                      ? OutlinedButton(
                          onPressed: () {},
                          style: OutlinedButton.styleFrom(
                            foregroundColor: AppColors.error,
                            side: const BorderSide(color: AppColors.error),
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          ),
                          child: Text('Cancel', style: GoogleFonts.sora(fontSize: 12, fontWeight: FontWeight.w600)),
                        )
                      : const SizedBox.shrink();

          if (compact) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Total: \u20B9${order.totalAmount.toStringAsFixed(0)}',
                  style: GoogleFonts.sora(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 12),
                actions,
              ],
            );
          }

          return Row(
            children: [
              Expanded(
                child: Text(
                  'Total: \u20B9${order.totalAmount.toStringAsFixed(0)}',
                  style: GoogleFonts.sora(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Flexible(child: actions),
            ],
          );
        },
      ),
    );
  }
}
