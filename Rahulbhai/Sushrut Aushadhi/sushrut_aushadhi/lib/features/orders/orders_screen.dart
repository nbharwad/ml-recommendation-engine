import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import 'package:url_launcher/url_launcher.dart';

import '../../core/constants/app_colors.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../core/widgets/order_card_widget.dart';
import '../../core/widgets/order_filter_chips.dart';
import '../../core/widgets/orders_loading_shimmer.dart';
import '../../models/order_model.dart';
import '../../providers/cart_provider.dart';
import '../../providers/orders_provider.dart';
import '../../services/connectivity_service.dart';
import '../../services/remote_config_service.dart';

final selectedOrderFilterProvider = StateProvider<String?>((ref) => null);

class OrdersScreen extends ConsumerStatefulWidget {
  const OrdersScreen({super.key});

  @override
  ConsumerState<OrdersScreen> createState() => _OrdersScreenState();
}

class _OrdersScreenState extends ConsumerState<OrdersScreen> {
  bool _isReordering = false;

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _onRefresh() async {
    ref.invalidate(ordersProvider);
    await Future.delayed(const Duration(milliseconds: 500));
  }

  Future<void> _handleReorder(OrderModel order) async {
    if (_isReordering) return;

    final isOnline = await ConnectivityService.checkConnection();
    if (!isOnline) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              const Icon(Icons.wifi_off, color: Colors.white, size: 16),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'No internet connection.\nPlease connect to reorder.',
                  style: GoogleFonts.sora(),
                ),
              ),
            ],
          ),
          backgroundColor: const Color(0xFFE53935),
          duration: const Duration(seconds: 4),
        ),
      );
      return;
    }

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
      final result =
          await ref.read(cartProvider.notifier).reorderFromOrder(order);

      if (!mounted) return;
      Navigator.of(context).pop();

      String message = '';
      Color backgroundColor = const Color(0xFF0F6E56);

      if (result.isSuccess) {
        message = '${result.totalAdded} item(s) added to cart!';
        backgroundColor = const Color(0xFF0F6E56);

        if (result.outOfStockItems.isNotEmpty) {
          message += '\n Out of stock: ${result.outOfStockItems.join(', ')}';
        }
        if (result.notFoundItems.isNotEmpty) {
          message += '\n Not available: ${result.notFoundItems.join(', ')}';
        }
      } else {
        message = 'Could not add any items to cart';
        backgroundColor = const Color(0xFFE53935);

        if (result.outOfStockItems.isNotEmpty) {
          message += '\n Out of stock: ${result.outOfStockItems.join(', ')}';
        }
        if (result.notFoundItems.isNotEmpty) {
          message += '\n Not available: ${result.notFoundItems.join(', ')}';
        }
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: backgroundColor,
          behavior: SnackBarBehavior.floating,
          duration: const Duration(seconds: 4),
          action: result.isSuccess
              ? SnackBarAction(
                  label: 'View Cart',
                  textColor: Colors.white,
                  onPressed: () => context.go('/cart'),
                )
              : null,
        ),
      );

      if (result.isSuccess && mounted) {
        context.go('/cart');
      }
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
          title: Text('Rate Order',
              style: GoogleFonts.sora(fontWeight: FontWeight.w600)),
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
                    onTap: () =>
                        setDialogState(() => selectedRating = starIndex),
                    child: Icon(
                      starIndex <= selectedRating
                          ? Icons.star_rounded
                          : Icons.star_outline_rounded,
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

  Future<void> _cancelOrder(OrderModel order) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Cancel Order',
            style: GoogleFonts.sora(fontWeight: FontWeight.bold)),
        content: Text(
          'Are you sure you want to cancel this order?',
          style: GoogleFonts.sora(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: Text('No',
                style: GoogleFonts.sora(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('Yes, Cancel',
                style: GoogleFonts.sora(color: AppColors.error)),
          ),
        ],
      ),
    );

    if (confirm != true || !mounted) return;

    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) throw Exception('Not authenticated');

      await FirebaseFirestore.instance
          .collection('orders')
          .doc(order.orderId)
          .update({
        'status': 'cancelled',
        'cancelReason': 'Cancelled by customer',
        'updatedAt': FieldValue.serverTimestamp(),
        'statusHistory': FieldValue.arrayUnion([
          {
            'status': 'cancelled',
            'timestamp': Timestamp.now(),
            'updatedBy': user.uid,
            'role': 'customer',
          }
        ]),
      });
      ref.invalidate(ordersProvider);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Order cancelled successfully'),
          backgroundColor: Color(0xFF0F6E56),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    }
  }

  Future<void> _callStore() async {
    final uri = Uri.parse('tel:${RemoteConfigService.storePhone}');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    }
  }

  @override
  Widget build(BuildContext context) {
    final ordersAsync = ref.watch(ordersProvider);
    final selectedFilter = ref.watch(selectedOrderFilterProvider);
    final user = FirebaseAuth.instance.currentUser;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        scrolledUnderElevation: 1,
        title: Text(
          'My Orders',
          style: GoogleFonts.sora(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        centerTitle: true,
        iconTheme: const IconThemeData(color: AppColors.textPrimary),
      ),
      body: ordersAsync.when(
        data: (orders) {
          if (user == null) {
            return _buildLoginPrompt();
          }

          final filteredOrders = _filterOrders(orders, selectedFilter);
          final filterCounts = _getFilterCounts(orders);

          return Column(
            children: [
              _buildFilterSection(selectedFilter, filterCounts),
              Expanded(
                child: filteredOrders.isEmpty
                    ? _buildEmptyState(selectedFilter)
                    : _buildOrdersList(filteredOrders),
              ),
            ],
          );
        },
        loading: () => _buildLoadingState(),
        error: (error, _) => _buildErrorState(error),
      ),
    );
  }

  Widget _buildFilterSection(String? selectedFilter, Map<String, int> counts) {
    return Container(
      color: Colors.white,
      child: Column(
        children: [
          const SizedBox(height: 8),
          OrderFilterChips(
            selectedFilter: selectedFilter,
            onFilterSelected: (filter) {
              HapticFeedback.selectionClick();
              ref.read(selectedOrderFilterProvider.notifier).state = filter;
            },
            counts: counts,
          ),
          const SizedBox(height: 4),
        ],
      ),
    );
  }

  Map<String, int> _getFilterCounts(List<OrderModel> orders) {
    return {
      'all': orders.length,
      'pending': orders.where((o) => o.status == OrderStatus.pending).length,
      'confirmed':
          orders.where((o) => o.status == OrderStatus.confirmed).length,
      'preparing':
          orders.where((o) => o.status == OrderStatus.preparing).length,
      'outForDelivery':
          orders.where((o) => o.status == OrderStatus.outForDelivery).length,
      'delivered':
          orders.where((o) => o.status == OrderStatus.delivered).length,
      'cancelled':
          orders.where((o) => o.status == OrderStatus.cancelled).length,
    };
  }

  Widget _buildLoadingState() {
    return Column(
      children: [
        _buildFilterSection(null, {}),
        const Expanded(
          child: OrdersLoadingList(itemCount: 5),
        ),
      ],
    );
  }

  Widget _buildErrorState(Object error) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              '⚠️',
              style: TextStyle(fontSize: 48),
            ),
            const SizedBox(height: 16),
            Text(
              'Something went wrong',
              style: GoogleFonts.sora(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              error.toString(),
              style: GoogleFonts.sora(
                fontSize: 14,
                color: AppColors.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () => ref.invalidate(ordersProvider),
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Retry'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
            ),
          ],
        ),
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

  Widget _buildEmptyState(String? filter) {
    final messages = _getEmptyStateMessages();
    final message = messages[filter ?? 'all'] ?? messages['all']!;

    return RefreshIndicator(
      color: AppColors.primary,
      backgroundColor: Colors.white,
      strokeWidth: 2.5,
      onRefresh: _onRefresh,
      child: CustomScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        slivers: [
          SliverFillRemaining(
            hasScrollBody: false,
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      message['emoji']!,
                      style: const TextStyle(fontSize: 64),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      message['title']!,
                      style: GoogleFonts.sora(
                        fontSize: 20,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textPrimary,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      message['subtitle']!,
                      style: GoogleFonts.sora(
                        fontSize: 14,
                        color: AppColors.textSecondary,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Map<String, Map<String, String>> _getEmptyStateMessages() {
    return {
      'all': {
        'emoji': '📦',
        'title': 'No Orders Yet',
        'subtitle':
            "You haven't placed any orders yet. Start shopping to see your orders here.",
      },
      'pending': {
        'emoji': '⏳',
        'title': 'No Pending Orders',
        'subtitle': "You don't have any orders waiting for confirmation.",
      },
      'confirmed': {
        'emoji': '✓',
        'title': 'No Confirmed Orders',
        'subtitle': "You don't have any confirmed orders.",
      },
      'preparing': {
        'emoji': '📋',
        'title': 'No Orders Being Prepared',
        'subtitle': "You don't have any orders currently being prepared.",
      },
      'outForDelivery': {
        'emoji': '🚚',
        'title': 'No Orders On The Way',
        'subtitle': "You don't have any orders out for delivery.",
      },
      'delivered': {
        'emoji': '✅',
        'title': 'No Delivered Orders',
        'subtitle': "You don't have any delivered orders yet.",
      },
      'cancelled': {
        'emoji': '❌',
        'title': 'No Cancelled Orders',
        'subtitle': "You don't have any cancelled orders.",
      },
    };
  }

  List<OrderModel> _filterOrders(List<OrderModel> orders, String? filter) {
    if (filter == null || filter == 'all') {
      return orders;
    }

    final status = OrderStatus.fromString(filter);
    return orders.where((o) => o.status == status).toList();
  }

  Widget _buildOrdersList(List<OrderModel> orders) {
    return RefreshIndicator(
      color: AppColors.primary,
      backgroundColor: Colors.white,
      strokeWidth: 2.5,
      onRefresh: _onRefresh,
      child: ListView.builder(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        itemCount: orders.length,
        itemBuilder: (context, index) {
          final order = orders[index];
          return Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: OrderCardWidget(
              order: order,
              onTap: () => context.push('/order/${order.orderId}'),
              onCancel: () => _cancelOrder(order),
              onReorder: () => _handleReorder(order),
              onRate: () => _showRatingDialog(order),
              onCallStore: _callStore,
              isReordering: _isReordering,
            ),
          );
        },
      ),
    );
  }
}
