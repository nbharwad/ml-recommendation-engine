import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../models/order_model.dart';
import '../constants/app_colors.dart';

class OrderStatusChip extends StatelessWidget {
  final OrderStatus status;
  final bool showIcon;

  const OrderStatusChip({
    super.key,
    required this.status,
    this.showIcon = true,
  });

  @override
  Widget build(BuildContext context) {
    final config = _getStatusConfig(status);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: config.backgroundColor.withOpacity(0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: config.backgroundColor.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (showIcon && config.icon != null) ...[
            Icon(
              config.icon,
              size: 14,
              color: config.backgroundColor,
            ),
            const SizedBox(width: 4),
          ],
          Text(
            config.label,
            style: GoogleFonts.sora(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: config.backgroundColor,
            ),
          ),
        ],
      ),
    );
  }

  _StatusConfig _getStatusConfig(OrderStatus status) {
    switch (status) {
      case OrderStatus.pending:
        return _StatusConfig(
          label: 'Pending',
          backgroundColor: AppColors.statusPending,
          icon: Icons.hourglass_top_rounded,
        );
      case OrderStatus.confirmed:
        return _StatusConfig(
          label: 'Confirmed',
          backgroundColor: AppColors.statusConfirmed,
          icon: Icons.check_circle_outline,
        );
      case OrderStatus.preparing:
        return _StatusConfig(
          label: 'Preparing',
          backgroundColor: AppColors.statusPreparing,
          icon: Icons.inventory_2_outlined,
        );
      case OrderStatus.outForDelivery:
        return _StatusConfig(
          label: 'Out for Delivery',
          backgroundColor: AppColors.statusOutForDelivery,
          icon: Icons.local_shipping_outlined,
        );
      case OrderStatus.delivered:
        return _StatusConfig(
          label: 'Delivered',
          backgroundColor: AppColors.statusDelivered,
          icon: Icons.task_alt,
        );
      case OrderStatus.cancelled:
        return _StatusConfig(
          label: 'Cancelled',
          backgroundColor: AppColors.statusCancelled,
          icon: Icons.cancel_outlined,
        );
    }
  }
}

class _StatusConfig {
  final String label;
  final Color backgroundColor;
  final IconData? icon;

  _StatusConfig({
    required this.label,
    required this.backgroundColor,
    required this.icon,
  });
}

class OrderStatusColor {
  static Color getBackgroundColor(OrderStatus status) {
    switch (status) {
      case OrderStatus.pending:
        return AppColors.statusPending;
      case OrderStatus.confirmed:
        return AppColors.statusConfirmed;
      case OrderStatus.preparing:
        return AppColors.statusPreparing;
      case OrderStatus.outForDelivery:
        return AppColors.statusOutForDelivery;
      case OrderStatus.delivered:
        return AppColors.statusDelivered;
      case OrderStatus.cancelled:
        return AppColors.statusCancelled;
    }
  }

  static Color getLightColor(OrderStatus status) {
    return getBackgroundColor(status).withOpacity(0.12);
  }
}
