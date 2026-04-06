import 'package:cached_network_image/cached_network_image.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../core/constants/app_colors.dart';
import '../../services/remote_config_service.dart';
import '../../core/utils/helpers.dart';
import '../../core/widgets/empty_state_widget.dart';
import '../../core/widgets/error_state_widget.dart';
import '../../models/order_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/orders_provider.dart';
import '../../services/whatsapp_service.dart';

// Responsive breakpoint — 2-column card grid above this width
const _kDesktopBreakpoint = 900.0;

class AdminOrdersScreen extends ConsumerStatefulWidget {
  const AdminOrdersScreen({super.key});

  @override
  ConsumerState<AdminOrdersScreen> createState() => _AdminOrdersScreenState();
}

class _AdminOrdersScreenState extends ConsumerState<AdminOrdersScreen>
    with AutomaticKeepAliveClientMixin {
  final _searchController = TextEditingController();
  final _scrollController = ScrollController();
  String _searchQuery = '';
  String _sortOrder = 'newest';
  String _selectedStatus = 'all';

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _scrollController.removeListener(_onScroll);
    _searchController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  bool get wantKeepAlive => true;

  void _onScroll() {
    final pos = _scrollController.position;
    if (pos.pixels >= pos.maxScrollExtent * 0.80) {
      final pageState = ref.read(adminOrdersPageProvider);
      if (!pageState.isLoadingMore && pageState.hasMore) {
        ref.read(adminOrdersPageProvider.notifier).loadNextPage();
      }
    }
  }

  Future<void> _onRefresh() async {
    await ref.read(adminOrdersPageProvider.notifier).refresh();
  }

  void _onStatusSelected(String status) {
    setState(() => _selectedStatus = status);
    ref.read(adminOrdersPageProvider.notifier).loadFirstPage();
  }

  void _onSearchChanged(String query) {
    setState(() {
      _searchQuery = query.toLowerCase();
    });
  }

  void _onSortChanged(String sortOrder) {
    setState(() {
      _sortOrder = sortOrder;
    });
  }

  List<OrderModel> _filterAndSortOrders(List<OrderModel> orders) {
    var filtered = orders.where((order) {
      if (_searchQuery.isEmpty) return true;
      return order.orderId.toLowerCase().contains(_searchQuery) ||
          order.userName.toLowerCase().contains(_searchQuery) ||
          order.userPhone.contains(_searchQuery);
    }).toList();

    switch (_sortOrder) {
      case 'oldest':
        filtered.sort((a, b) => a.createdAt.compareTo(b.createdAt));
        break;
      case 'amount_high':
        filtered.sort((a, b) => b.totalAmount.compareTo(a.totalAmount));
        break;
      case 'amount_low':
        filtered.sort((a, b) => a.totalAmount.compareTo(b.totalAmount));
        break;
      case 'newest':
      default:
        filtered.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    }

    return filtered;
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final pageState = ref.watch(adminOrdersPageProvider);
    final stats = pageState.isInitialLoading
        ? _AdminOrderStats.empty()
        : _AdminOrderStats.fromOrders(pageState.orders);

    return RefreshIndicator(
      color: Colors.white,
      backgroundColor: AppColors.primary,
      onRefresh: _onRefresh,
      child: CustomScrollView(
        controller: _scrollController,
        physics: const AlwaysScrollableScrollPhysics(),
        slivers: [
          // ── M3 Collapsing App Bar ──────────────────────────────────────
          _buildSliverAppBar(stats),

          // ── Pinned: Search + Sort + Filter chips ──────────────────────
          SliverPersistentHeader(
            pinned: true,
            delegate: _StickySearchDelegate(
              searchController: _searchController,
              sortOrder: _sortOrder,
              selectedStatus: _selectedStatus,
              stats: stats,
              onSearchChanged: _onSearchChanged,
              onSortChanged: _onSortChanged,
              onStatusSelected: _onStatusSelected,
            ),
          ),

          // ── Orders list ───────────────────────────────────────────────
          _buildOrdersSliver(pageState),
        ],
      ),
    );
  }

  // ── Sliver App Bar ─────────────────────────────────────────────────────────

  SliverAppBar _buildSliverAppBar(_AdminOrderStats stats) {
    return SliverAppBar(
      expandedHeight: 148,
      collapsedHeight: 56,
      pinned: true,
      stretch: true,
      backgroundColor: AppColors.primary,
      elevation: 0,
      automaticallyImplyLeading: false,
      flexibleSpace: LayoutBuilder(
        builder: (context, constraints) {
          // How collapsed are we? 0 = fully expanded, 1 = fully collapsed
          final expandedHeight = 148.0 + MediaQuery.of(context).padding.top;
          final collapsedHeight = 56.0 + MediaQuery.of(context).padding.top;
          final t = ((expandedHeight - constraints.maxHeight) /
                  (expandedHeight - collapsedHeight))
              .clamp(0.0, 1.0);

          return Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Color(0xFF0A5240), Color(0xFF1D9E75)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Stack(
                children: [
                  // Expanded content — fades out as collapsed
                  Opacity(
                    opacity: (1 - t * 2).clamp(0.0, 1.0),
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(20, 12, 20, 12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: Text(
                                  'Medicine Orders',
                                  style: GoogleFonts.sora(
                                    color: Colors.white,
                                    fontSize: 22,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                              _buildHeaderActions(),
                            ],
                          ),
                          const SizedBox(height: 16),
                          _buildInlineStats(stats),
                        ],
                      ),
                    ),
                  ),
                  // Collapsed title — fades in
                  Opacity(
                    opacity: ((t - 0.5) * 2).clamp(0.0, 1.0),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 20),
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                'Medicine Orders',
                                style: GoogleFonts.sora(
                                  color: Colors.white,
                                  fontSize: 17,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                            _buildHeaderActions(),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildHeaderActions() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Prescriptions shortcut
        _GlassButton(
          onTap: () => context.push('/admin/prescriptions'),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.description_outlined,
                  size: 15, color: Colors.white),
              const SizedBox(width: 5),
              Text(
                'Rx',
                style: GoogleFonts.sora(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        // Refresh
        _GlassButton(
          onTap: _onRefresh,
          child:
              const Icon(Icons.refresh_rounded, color: Colors.white, size: 20),
        ),
      ],
    );
  }

  /// Compact inline stats strip shown in expanded app bar
  Widget _buildInlineStats(_AdminOrderStats stats) {
    final items = [
      (
        label: '${stats.pendingCount}',
        sub: 'Pending',
        color: AppColors.statusPending
      ),
      (
        label: '${stats.confirmedCount}',
        sub: 'Confirmed',
        color: AppColors.statusConfirmed
      ),
      (
        label: '${stats.deliveredTodayCount}',
        sub: 'Delivered Today',
        color: AppColors.statusDelivered
      ),
      (
        label:
            'Rs ${stats.todayRevenue < 1000 ? stats.todayRevenue.toStringAsFixed(0) : '${(stats.todayRevenue / 1000).toStringAsFixed(1)}k'}',
        sub: "Today's Rev",
        color: const Color(0xFF80CBC4),
      ),
    ];

    return Row(
      children: items
          .map(
            (item) => Expanded(
              child: _StatPill(
                value: item.label,
                label: item.sub,
                valueColor: item.color,
              ),
            ),
          )
          .toList(),
    );
  }

  // ── Orders Sliver ──────────────────────────────────────────────────────────

  Widget _buildOrdersSliver(AdminOrdersPageState pageState) {
    // Auth loading state
    if (pageState.authStatus == AdminAuthStatus.loading ||
        pageState.isInitialLoading) {
      return const SliverFillRemaining(
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(color: AppColors.primary),
              SizedBox(height: 16),
              Text('Verifying admin access...'),
            ],
          ),
        ),
      );
    }

    // Auth error state
    if (pageState.authStatus == AdminAuthStatus.error) {
      return SliverFillRemaining(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: ErrorStateWidget(
              message:
                  'Could not verify admin access.\n${pageState.errorMessage ?? ''}',
              onRetry: _onRefresh,
            ),
          ),
        ),
      );
    }

    // Not admin state
    if (pageState.authStatus == AdminAuthStatus.notAdmin) {
      return const SliverFillRemaining(
        child: Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: EmptyStateWidget(
              emoji: '🔒',
              title: 'Access Denied',
              subtitle: 'You do not have permission to view this page.',
            ),
          ),
        ),
      );
    }

    // Firestore error states
    if (pageState.errorType != PageErrorType.none && pageState.orders.isEmpty) {
      String message;
      switch (pageState.errorType) {
        case PageErrorType.permissionDenied:
          message =
              'Permission denied. Ensure your account has admin access and try refreshing.';
          break;
        case PageErrorType.queryPrecondition:
          message = 'Query configuration error. Please contact support.';
          break;
        case PageErrorType.network:
          message =
              'Network error. Please check your connection and try again.';
          break;
        case PageErrorType.authResolution:
          message =
              'Could not verify admin access.\n${pageState.errorMessage ?? ''}';
          break;
        case PageErrorType.unknown:
        default:
          message = 'Could not load orders.\n${pageState.errorMessage ?? ''}';
      }
      return SliverFillRemaining(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: ErrorStateWidget(
              message: message,
              onRetry: _onRefresh,
            ),
          ),
        ),
      );
    }

    var filteredOrders = _selectedStatus == 'all'
        ? pageState.orders
        : pageState.orders
            .where((o) => o.status.name == _selectedStatus)
            .toList();
    filteredOrders = _filterAndSortOrders(filteredOrders);

    if (filteredOrders.isEmpty) {
      return SliverFillRemaining(
        child: _buildEmptyState(_selectedStatus),
      );
    }

    return SliverPadding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      sliver: LayoutBuilder(
        builder: (context, constraints) {
          final width = MediaQuery.of(context).size.width;
          final extraItem = pageState.isLoadingMore ? 1 : 0;
          if (width >= _kDesktopBreakpoint) {
            // 2-column grid for large screens
            return SliverGrid.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                childAspectRatio: 1.05,
              ),
              itemCount: filteredOrders.length + extraItem,
              itemBuilder: (context, index) {
                if (index == filteredOrders.length) {
                  return _buildLoadMoreIndicator();
                }
                return _buildOrderCard(filteredOrders[index]);
              },
            );
          }
          // Single column for phone / small tablet
          return SliverList.separated(
            itemCount: filteredOrders.length + extraItem,
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemBuilder: (context, index) {
              if (index == filteredOrders.length) {
                return _buildLoadMoreIndicator();
              }
              return _buildOrderCard(filteredOrders[index]);
            },
          );
        },
      ),
    );
  }

  Widget _buildLoadMoreIndicator() {
    return const Padding(
      padding: EdgeInsets.symmetric(vertical: 24),
      child: Center(child: CircularProgressIndicator(color: AppColors.primary)),
    );
  }

  Widget _buildEmptyState(String status) {
    return switch (status) {
      'pending' => const EmptyStateWidget(
          emoji: '⏳',
          title: 'No Pending Orders',
          subtitle: 'There are no orders waiting for confirmation.',
        ),
      'confirmed' => const EmptyStateWidget(
          emoji: '📋',
          title: 'No Confirmed Orders',
          subtitle: 'There are no confirmed orders to prepare.',
        ),
      'delivered' => const EmptyStateWidget(
          emoji: '✅',
          title: 'No Delivered Orders',
          subtitle: 'There are no delivered orders yet.',
        ),
      _ => const EmptyStateWidget(
          emoji: '📦',
          title: 'No Orders',
          subtitle: 'There are no orders yet.',
        ),
    };
  }

  // ── Order Card ─────────────────────────────────────────────────────────────

  Widget _buildOrderCard(OrderModel order) {
    final statusColor = Helpers.getStatusColor(order.status);
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE8ECE7)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Thin colored top border per status
          Container(
            height: 3,
            decoration: BoxDecoration(
              color: statusColor,
              borderRadius:
                  const BorderRadius.vertical(top: Radius.circular(16)),
            ),
          ),
          _buildCardHeader(order),
          _buildCustomerSection(order),
          _buildItemsList(order),
          if (order.status != OrderStatus.delivered &&
              order.status != OrderStatus.cancelled)
            _buildStatusUpdateRow(order),
          _buildCardFooter(order),
        ],
      ),
    );
  }

  Widget _buildCardHeader(OrderModel order) {
    final statusColor = Helpers.getStatusColor(order.status);
    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 10),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'SA-${order.orderId.substring(0, 4).toUpperCase()}',
                  style: GoogleFonts.sora(
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  _getTimeAgo(order.createdAt),
                  style: GoogleFonts.sora(
                    color: AppColors.textSecondary,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: statusColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(999),
              border: Border.all(color: statusColor.withValues(alpha: 0.3)),
            ),
            child: Text(
              order.status.displayName,
              style: GoogleFonts.sora(
                color: statusColor,
                fontWeight: FontWeight.w700,
                fontSize: 11,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCustomerSection(OrderModel order) {
    final initials = order.userName.isNotEmpty
        ? order.userName
            .split(' ')
            .map((s) => s.isNotEmpty ? s[0] : '')
            .take(2)
            .join()
            .toUpperCase()
        : 'CU';

    final actionButtons = Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _ActionIconButton(
          icon: Icons.call_rounded,
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          onTap: () => _launchUrl('tel:${order.userPhone}'),
        ),
        _ActionIconButton(
          icon: Icons.chat_bubble_outline_rounded,
          backgroundColor: const Color(0xFFE8F5E9),
          foregroundColor: const Color(0xFF1B8E3E),
          onTap: () async {
            final itemNames =
                order.items.map((item) => item.medicineName).toList();
            try {
              await WhatsAppService.sendOrderUpdate(
                customerPhone: order.userPhone,
                orderId: order.orderId,
                customerName: order.userName,
                status: order.status.name,
                totalAmount: order.totalAmount,
                itemNames: itemNames,
                storePhone: RemoteConfigService.storePhone,
              );
            } catch (_) {
              if (!mounted) return;
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('WhatsApp is not available on this device'),
                  backgroundColor: Colors.orange,
                ),
              );
            }
          },
        ),
        if (order.prescriptionUrl != null && order.prescriptionUrl!.isNotEmpty)
          _ActionIconButton(
            icon: Icons.description_outlined,
            backgroundColor: const Color(0xFFF3E5F5),
            foregroundColor: const Color(0xFF7B1FA2),
            onTap: () => _showPrescriptionViewer(order.prescriptionUrl!),
          ),
      ],
    );

    return Container(
      padding: const EdgeInsets.fromLTRB(14, 0, 14, 12),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0xFFF0F4F0))),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final customerInfo = Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  order.userName.isNotEmpty ? order.userName : 'Customer',
                  style: GoogleFonts.sora(
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  order.userPhone,
                  style: GoogleFonts.sora(
                    color: AppColors.textSecondary,
                    fontSize: 11,
                  ),
                ),
                if (!order.deliveryAddress.isEmpty) ...[
                  const SizedBox(height: 3),
                  Row(
                    children: [
                      const Icon(Icons.location_on_outlined,
                          size: 12, color: AppColors.textSecondary),
                      const SizedBox(width: 3),
                      Expanded(
                        child: Text(
                          order.deliveryAddress.toDisplayString(),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.sora(
                            color: AppColors.textSecondary,
                            fontSize: 11,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          );

          if (constraints.maxWidth < 420) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CircleAvatar(
                      radius: 18,
                      backgroundColor: AppColors.primary,
                      child: Text(
                        initials,
                        style: GoogleFonts.sora(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    customerInfo,
                  ],
                ),
                const SizedBox(height: 10),
                actionButtons,
              ],
            );
          }

          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CircleAvatar(
                radius: 18,
                backgroundColor: AppColors.primary,
                child: Text(
                  initials,
                  style: GoogleFonts.sora(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              customerInfo,
              const SizedBox(width: 8),
              Flexible(
                  child: Align(
                      alignment: Alignment.topRight, child: actionButtons)),
            ],
          );
        },
      ),
    );
  }

  Widget _buildItemsList(OrderModel order) {
    final displayItems = order.items.take(3).toList();
    final remainingCount = order.items.length - displayItems.length;

    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 3,
                height: 12,
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 6),
              Text(
                '${order.itemCount} item${order.itemCount == 1 ? '' : 's'}',
                style: GoogleFonts.sora(
                  fontSize: 11,
                  color: AppColors.textSecondary,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.2,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ...displayItems.map(
            (item) => Padding(
              padding: const EdgeInsets.only(bottom: 5),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 5,
                    height: 5,
                    margin: const EdgeInsets.only(top: 7),
                    decoration: const BoxDecoration(
                      color: AppColors.primary,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      '${item.medicineName} × ${item.quantity}',
                      style: GoogleFonts.sora(fontSize: 12, height: 1.4),
                    ),
                  ),
                  Text(
                    'Rs ${item.subtotal.toStringAsFixed(0)}',
                    style: GoogleFonts.sora(
                      fontSize: 11,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (remainingCount > 0)
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Text(
                '+$remainingCount more item${remainingCount == 1 ? '' : 's'}',
                style: GoogleFonts.sora(
                  color: AppColors.primary,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildStatusUpdateRow(OrderModel order) {
    final nextStatuses = _getNextStatuses(order.status);
    if (nextStatuses.isEmpty) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.fromLTRB(14, 8, 14, 10),
      decoration: const BoxDecoration(
        border: Border(
          top: BorderSide(color: Color(0xFFF0F4F0)),
          bottom: BorderSide(color: Color(0xFFF0F4F0)),
        ),
      ),
      child: Row(
        children: [
          Text(
            'Move to:',
            style: GoogleFonts.sora(
              color: AppColors.textSecondary,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Wrap(
              spacing: 8,
              runSpacing: 6,
              children: nextStatuses
                  .map(
                    (status) => _StatusChipButton(
                      label: status.displayName,
                      color: Helpers.getStatusColor(status),
                      onTap: () => _updateOrderStatus(order.orderId, status),
                    ),
                  )
                  .toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCardFooter(OrderModel order) {
    final canAct = order.status != OrderStatus.cancelled &&
        order.status != OrderStatus.delivered;

    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final totalBlock = Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Rs ${order.totalAmount.toStringAsFixed(0)}',
                style: GoogleFonts.sora(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                  color: AppColors.primary,
                ),
              ),
              if (order.prescriptionUrl != null &&
                  order.prescriptionUrl!.isNotEmpty)
                Text(
                  '📎 Prescription attached',
                  style: GoogleFonts.sora(
                    fontSize: 10,
                    color: const Color(0xFF7B1FA2),
                    fontWeight: FontWeight.w500,
                  ),
                ),
            ],
          );

          if (!canAct) return totalBlock;

          final actions = Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              OutlinedButton(
                onPressed: () => _showRejectConfirmDialog(order.orderId),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.error,
                  side:
                      BorderSide(color: AppColors.error.withValues(alpha: 0.6)),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: Text('Reject', style: GoogleFonts.sora(fontSize: 12)),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: () => _showConfirmDialog(order),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: Text(
                  order.status == OrderStatus.pending ? 'Confirm' : 'Update',
                  style: GoogleFonts.sora(
                      fontSize: 12, fontWeight: FontWeight.w600),
                ),
              ),
            ],
          );

          if (constraints.maxWidth < 380) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [totalBlock, const SizedBox(height: 10), actions],
            );
          }

          return Row(
            children: [
              Expanded(child: totalBlock),
              actions,
            ],
          );
        },
      ),
    );
  }

  // ── Status logic ───────────────────────────────────────────────────────────

  List<OrderStatus> _getNextStatuses(OrderStatus current) {
    switch (current) {
      case OrderStatus.pending:
        return [OrderStatus.confirmed, OrderStatus.cancelled];
      case OrderStatus.confirmed:
        return [OrderStatus.preparing, OrderStatus.outForDelivery];
      case OrderStatus.preparing:
        return [OrderStatus.outForDelivery];
      case OrderStatus.outForDelivery:
        return [OrderStatus.delivered];
      case OrderStatus.delivered:
      case OrderStatus.cancelled:
        return [];
    }
  }

  Future<void> _updateOrderStatus(String orderId, OrderStatus newStatus,
      {String? updatedBy}) async {
    try {
      final orderDoc = await FirebaseFirestore.instance
          .collection('orders')
          .doc(orderId)
          .get();
      if (!orderDoc.exists) return;

      final orderData = orderDoc.data()!;
      final currentStatus = orderData['status'] ?? 'pending';
      final userId = orderData['userId'] ?? '';

      final existingHistory = (orderData['statusHistory'] as List?)
              ?.map((e) => Map<String, dynamic>.from(e as Map))
              .toList() ??
          [];
      existingHistory.add({
        'status': newStatus.name,
        'timestamp': FieldValue.serverTimestamp(),
        'updatedBy': updatedBy ?? 'admin',
        'role': 'admin',
      });

      final updateData = <String, dynamic>{
        'status': newStatus.name,
        'updatedAt': FieldValue.serverTimestamp(),
        'statusHistory': existingHistory,
      };

      if (newStatus == OrderStatus.delivered) {
        updateData['deliveredAt'] = FieldValue.serverTimestamp();
      }

      await FirebaseFirestore.instance
          .collection('orders')
          .doc(orderId)
          .update(updateData);

      if (userId.isNotEmpty && currentStatus != newStatus.name) {
        await FirebaseFirestore.instance.collection('notifications').add({
          'userId': userId,
          'title': 'Order Status Updated',
          'body': 'Your order has been ${newStatus.displayName.toLowerCase()}.',
          'type': 'order_status',
          'orderId': orderId,
          'isRead': false,
          'createdAt': FieldValue.serverTimestamp(),
        });
      }

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Order status updated to ${newStatus.displayName}',
            style: GoogleFonts.sora(),
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content:
              Text('Failed to update order: $error', style: GoogleFonts.sora()),
        ),
      );
    }
  }

  void _showConfirmDialog(OrderModel order) {
    final nextStatuses = _getNextStatuses(order.status);
    if (nextStatuses.isEmpty) return;

    showDialog<void>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(
          'Update Status',
          style: GoogleFonts.sora(fontWeight: FontWeight.bold),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: nextStatuses
              .map(
                (status) => ListTile(
                  leading: CircleAvatar(
                    radius: 10,
                    backgroundColor: Helpers.getStatusColor(status),
                  ),
                  title: Text(status.displayName, style: GoogleFonts.sora()),
                  onTap: () {
                    Navigator.of(dialogContext).pop();
                    _updateOrderStatus(order.orderId, status);
                  },
                ),
              )
              .toList(),
        ),
      ),
    );
  }

  void _showRejectConfirmDialog(String orderId) {
    showDialog<void>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(
          'Reject Order',
          style: GoogleFonts.sora(fontWeight: FontWeight.bold),
        ),
        content: Text(
          'Are you sure you want to reject this order?',
          style: GoogleFonts.sora(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: Text(
              'Cancel',
              style: GoogleFonts.sora(color: AppColors.textSecondary),
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(dialogContext).pop();
              _updateOrderStatus(orderId, OrderStatus.cancelled);
            },
            child:
                Text('Reject', style: GoogleFonts.sora(color: AppColors.error)),
          ),
        ],
      ),
    );
  }

  void _showPrescriptionViewer(String url) {
    showDialog<void>(
      context: context,
      builder: (dialogContext) => Dialog(
        backgroundColor: Colors.black,
        insetPadding: EdgeInsets.zero,
        child: Stack(
          children: [
            Positioned.fill(
              child: InteractiveViewer(
                minScale: 0.5,
                maxScale: 4,
                child: CachedNetworkImage(
                  imageUrl: url,
                  fit: BoxFit.contain,
                  placeholder: (_, __) => const Center(
                      child: CircularProgressIndicator(color: Colors.white)),
                  errorWidget: (_, __, ___) => const Center(
                    child:
                        Icon(Icons.broken_image, color: Colors.white, size: 64),
                  ),
                ),
              ),
            ),
            Positioned(
              top: MediaQuery.of(dialogContext).padding.top + 8,
              right: 8,
              child: IconButton(
                onPressed: () => Navigator.of(dialogContext).pop(),
                icon: const Icon(Icons.close, color: Colors.white, size: 28),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  String _getTimeAgo(DateTime dateTime) {
    final diff = DateTime.now().difference(dateTime);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays == 1) return 'Yesterday';
    return '${diff.inDays}d ago';
  }
}

// ── Sticky Search + Filter Delegate ─────────────────────────────────────────

class _StickySearchDelegate extends SliverPersistentHeaderDelegate {
  const _StickySearchDelegate({
    required this.searchController,
    required this.sortOrder,
    required this.selectedStatus,
    required this.stats,
    required this.onSearchChanged,
    required this.onSortChanged,
    required this.onStatusSelected,
  });

  final TextEditingController searchController;
  final String sortOrder;
  final String selectedStatus;
  final _AdminOrderStats stats;
  final ValueChanged<String> onSearchChanged;
  final ValueChanged<String> onSortChanged;
  final ValueChanged<String> onStatusSelected;

  static const double _height = 100;

  @override
  double get minExtent => _height;
  @override
  double get maxExtent => _height;

  @override
  bool shouldRebuild(_StickySearchDelegate old) =>
      old.sortOrder != sortOrder ||
      old.selectedStatus != selectedStatus ||
      old.stats != stats;

  @override
  Widget build(
      BuildContext context, double shrinkOffset, bool overlapsContent) {
    final chips = [
      (
        status: 'all',
        label: 'All',
        count: stats.totalCount,
        color: AppColors.primary
      ),
      (
        status: 'pending',
        label: 'Pending',
        count: stats.pendingCount,
        color: AppColors.statusPending
      ),
      (
        status: 'confirmed',
        label: 'Confirmed',
        count: stats.confirmedCount,
        color: AppColors.statusConfirmed
      ),
      (
        status: 'preparing',
        label: 'Preparing',
        count: stats.preparingCount,
        color: AppColors.statusPreparing
      ),
      (
        status: 'outForDelivery',
        label: 'Out for Delivery',
        count: stats.outForDeliveryCount,
        color: AppColors.statusOutForDelivery
      ),
      (
        status: 'delivered',
        label: 'Delivered',
        count: stats.deliveredCount,
        color: AppColors.statusDelivered
      ),
    ];

    return Material(
      color: AppColors.background,
      elevation: overlapsContent ? 2 : 0,
      shadowColor: Colors.black12,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Search + Sort row
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 10, 16, 6),
            child: Row(
              children: [
                Expanded(
                  child: Container(
                    height: 42,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFFE2E8E2)),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.04),
                          blurRadius: 4,
                          offset: const Offset(0, 1),
                        ),
                      ],
                    ),
                    child: TextField(
                      controller: searchController,
                      onChanged: onSearchChanged,
                      decoration: InputDecoration(
                        hintText: 'Search by name, ID or phone…',
                        hintStyle: GoogleFonts.sora(
                          color: AppColors.textSecondary,
                          fontSize: 12,
                        ),
                        prefixIcon: const Icon(
                          Icons.search_rounded,
                          color: AppColors.textSecondary,
                          size: 18,
                        ),
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 11),
                      ),
                      style: GoogleFonts.sora(fontSize: 13),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Container(
                  height: 42,
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFE2E8E2)),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.04),
                        blurRadius: 4,
                        offset: const Offset(0, 1),
                      ),
                    ],
                  ),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: sortOrder,
                      icon: const Icon(Icons.unfold_more_rounded,
                          color: AppColors.textSecondary, size: 18),
                      style: GoogleFonts.sora(
                          fontSize: 12, color: AppColors.textPrimary),
                      items: const [
                        DropdownMenuItem(
                            value: 'newest', child: Text('Newest')),
                        DropdownMenuItem(
                            value: 'oldest', child: Text('Oldest')),
                        DropdownMenuItem(
                            value: 'amount_high', child: Text('Rs ↑')),
                        DropdownMenuItem(
                            value: 'amount_low', child: Text('Rs ↓')),
                      ],
                      onChanged: (v) {
                        if (v != null) onSortChanged(v);
                      },
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Filter chips
          SizedBox(
            height: 36,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              itemCount: chips.length,
              separatorBuilder: (_, __) => const SizedBox(width: 7),
              itemBuilder: (context, i) {
                final c = chips[i];
                final isSelected = selectedStatus == c.status;
                return GestureDetector(
                  onTap: () => onStatusSelected(c.status),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    curve: Curves.easeInOut,
                    padding:
                        const EdgeInsets.symmetric(horizontal: 11, vertical: 5),
                    decoration: BoxDecoration(
                      color: isSelected ? c.color : Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: isSelected ? c.color : const Color(0xFFDDE3DD),
                      ),
                      boxShadow: isSelected
                          ? [
                              BoxShadow(
                                color: c.color.withValues(alpha: 0.25),
                                blurRadius: 6,
                                offset: const Offset(0, 2),
                              )
                            ]
                          : null,
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          c.label,
                          style: GoogleFonts.sora(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: isSelected
                                ? Colors.white
                                : AppColors.textSecondary,
                          ),
                        ),
                        if (c.count > 0) ...[
                          const SizedBox(width: 5),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 5, vertical: 1),
                            decoration: BoxDecoration(
                              color: isSelected
                                  ? Colors.white.withValues(alpha: 0.3)
                                  : c.color.withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(999),
                            ),
                            child: Text(
                              '${c.count}',
                              style: GoogleFonts.sora(
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                color: isSelected ? Colors.white : c.color,
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 6),
        ],
      ),
    );
  }
}

// ── Small helper widgets ─────────────────────────────────────────────────────

/// Frosted-glass style button for the app bar header
class _GlassButton extends StatelessWidget {
  const _GlassButton({required this.onTap, required this.child});

  final VoidCallback onTap;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.white.withValues(alpha: 0.25)),
        ),
        child: child,
      ),
    );
  }
}

/// Compact stat pill shown in expanded app bar
class _StatPill extends StatelessWidget {
  const _StatPill({
    required this.value,
    required this.label,
    required this.valueColor,
  });

  final String value;
  final String label;
  final Color valueColor;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          value,
          style: GoogleFonts.sora(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.sora(
            fontSize: 10,
            color: Colors.white.withValues(alpha: 0.72),
            height: 1.2,
          ),
        ),
      ],
    );
  }
}

/// Compact status-colored chip button in the order card "Move to" row
class _StatusChipButton extends StatelessWidget {
  const _StatusChipButton({
    required this.label,
    required this.color,
    required this.onTap,
  });

  final String label;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(999),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: color.withValues(alpha: 0.4)),
        ),
        child: Text(
          label,
          style: GoogleFonts.sora(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
    );
  }
}

/// Action icon button (call / WhatsApp / prescription)
class _ActionIconButton extends StatelessWidget {
  const _ActionIconButton({
    required this.icon,
    required this.backgroundColor,
    required this.foregroundColor,
    required this.onTap,
  });

  final IconData icon;
  final Color backgroundColor;
  final Color foregroundColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(10),
      onTap: onTap,
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          color: backgroundColor,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(icon, size: 18, color: foregroundColor),
      ),
    );
  }
}

// ── Data class ───────────────────────────────────────────────────────────────

class _AdminOrderStats {
  const _AdminOrderStats({
    required this.totalCount,
    required this.pendingCount,
    required this.confirmedCount,
    required this.preparingCount,
    required this.outForDeliveryCount,
    required this.deliveredCount,
    required this.deliveredTodayCount,
    required this.todayRevenue,
  });

  final int totalCount;
  final int pendingCount;
  final int confirmedCount;
  final int preparingCount;
  final int outForDeliveryCount;
  final int deliveredCount;
  final int deliveredTodayCount;
  final double todayRevenue;

  factory _AdminOrderStats.fromOrders(List<OrderModel> orders) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);

    var pendingCount = 0;
    var confirmedCount = 0;
    var preparingCount = 0;
    var outForDeliveryCount = 0;
    var deliveredCount = 0;
    var deliveredTodayCount = 0;
    var todayRevenue = 0.0;

    for (final order in orders) {
      switch (order.status) {
        case OrderStatus.pending:
          pendingCount++;
        case OrderStatus.confirmed:
          confirmedCount++;
        case OrderStatus.preparing:
          preparingCount++;
        case OrderStatus.outForDelivery:
          outForDeliveryCount++;
        case OrderStatus.delivered:
          deliveredCount++;
          final deliveredDate = order.deliveredAt ?? order.createdAt;
          final orderDay = DateTime(
            deliveredDate.year,
            deliveredDate.month,
            deliveredDate.day,
          );
          if (orderDay == today) {
            deliveredTodayCount++;
            todayRevenue += order.totalAmount;
          }
        case OrderStatus.cancelled:
          break;
      }
    }

    return _AdminOrderStats(
      totalCount: orders.length,
      pendingCount: pendingCount,
      confirmedCount: confirmedCount,
      preparingCount: preparingCount,
      outForDeliveryCount: outForDeliveryCount,
      deliveredCount: deliveredCount,
      deliveredTodayCount: deliveredTodayCount,
      todayRevenue: todayRevenue,
    );
  }

  static _AdminOrderStats empty() {
    return const _AdminOrderStats(
      totalCount: 0,
      pendingCount: 0,
      confirmedCount: 0,
      preparingCount: 0,
      outForDeliveryCount: 0,
      deliveredCount: 0,
      deliveredTodayCount: 0,
      todayRevenue: 0,
    );
  }
}
