import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../models/order_model.dart';
import '../constants/app_colors.dart';

class OrderTimelineWidget extends StatelessWidget {
  final OrderStatus currentStatus;
  final bool isExpanded;

  const OrderTimelineWidget({
    super.key,
    required this.currentStatus,
    this.isExpanded = false,
  });

  @override
  Widget build(BuildContext context) {
    final steps = _getSteps();
    final currentIndex = _getCurrentIndex();

    return LayoutBuilder(
      builder: (context, constraints) {
        final availableWidth = constraints.maxWidth;
        final stepWidth = availableWidth / steps.length;

        return SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          physics: const BouncingScrollPhysics(),
          child: Row(
            children: List.generate(steps.length, (index) {
              final step = steps[index];
              final isCompleted = index < currentIndex;
              final isCurrent = index == currentIndex;
              final isLast = index == steps.length - 1;

              return SizedBox(
                width: isLast ? stepWidth : stepWidth + 16,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Flexible(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: isCompleted || isCurrent
                                  ? AppColors.primary
                                  : Colors.grey.shade300,
                              boxShadow: isCurrent
                                  ? [
                                      BoxShadow(
                                        color:
                                            AppColors.primary.withOpacity(0.4),
                                        blurRadius: 8,
                                        spreadRadius: 2,
                                      ),
                                    ]
                                  : null,
                            ),
                            child: Center(
                              child: isCompleted
                                  ? const Icon(
                                      Icons.check,
                                      color: Colors.white,
                                      size: 14,
                                    )
                                  : Text(
                                      '${index + 1}',
                                      style: GoogleFonts.sora(
                                        fontSize: 10,
                                        fontWeight: FontWeight.w600,
                                        color: isCurrent
                                            ? Colors.white
                                            : AppColors.textSecondary,
                                      ),
                                    ),
                            ),
                          ),
                          const SizedBox(height: 6),
                          SizedBox(
                            width: 60,
                            child: Text(
                              step.label,
                              textAlign: TextAlign.center,
                              style: GoogleFonts.sora(
                                fontSize: 10,
                                color: isCurrent
                                    ? AppColors.primary
                                    : AppColors.textSecondary,
                                fontWeight: isCurrent
                                    ? FontWeight.w600
                                    : FontWeight.normal,
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (!isLast)
                      Flexible(
                        child: Container(
                          height: 2,
                          margin: const EdgeInsets.only(bottom: 24),
                          child: Container(
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  index < currentIndex
                                      ? AppColors.primary
                                      : Colors.grey.shade300,
                                  (index + 1) < currentIndex
                                      ? AppColors.primary
                                      : Colors.grey.shade300,
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              );
            }),
          ),
        );
      },
    );
  }

  List<_Step> _getSteps() {
    return [
      _Step(
        status: OrderStatus.pending,
        label: 'Ordered',
      ),
      _Step(
        status: OrderStatus.confirmed,
        label: 'Confirmed',
      ),
      _Step(
        status: OrderStatus.preparing,
        label: 'Preparing',
      ),
      _Step(
        status: OrderStatus.outForDelivery,
        label: 'On the way',
      ),
      _Step(
        status: OrderStatus.delivered,
        label: 'Delivered',
      ),
    ];
  }

  int _getCurrentIndex() {
    final statuses = [
      OrderStatus.pending,
      OrderStatus.confirmed,
      OrderStatus.preparing,
      OrderStatus.outForDelivery,
      OrderStatus.delivered,
    ];
    final index = statuses.indexOf(currentStatus);
    return index >= 0 ? index : 0;
  }
}

class _Step {
  final OrderStatus status;
  final String label;

  _Step({
    required this.status,
    required this.label,
  });
}

class CompactTimelineWidget extends StatelessWidget {
  final OrderStatus currentStatus;

  const CompactTimelineWidget({
    super.key,
    required this.currentStatus,
  });

  @override
  Widget build(BuildContext context) {
    final statuses = [
      OrderStatus.pending,
      OrderStatus.confirmed,
      OrderStatus.preparing,
      OrderStatus.outForDelivery,
      OrderStatus.delivered,
    ];
    final currentIndex = statuses.indexOf(currentStatus);
    final displayIndex = currentIndex.clamp(0, statuses.length - 1);

    return LayoutBuilder(
      builder: (context, constraints) {
        return Row(
          children: List.generate(statuses.length, (index) {
            final isCompleted = index <= displayIndex;
            final isCurrent = index == displayIndex;

            return Expanded(
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 10,
                          height: 10,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: isCompleted
                                ? AppColors.primary
                                : Colors.grey.shade300,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          _getShortLabel(statuses[index]),
                          textAlign: TextAlign.center,
                          style: GoogleFonts.sora(
                            fontSize: 8,
                            color: isCurrent
                                ? AppColors.primary
                                : AppColors.textSecondary,
                            fontWeight:
                                isCurrent ? FontWeight.w600 : FontWeight.normal,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                  if (index < statuses.length - 1)
                    Expanded(
                      child: Container(
                        height: 2,
                        color: (index + 1) <= displayIndex
                            ? AppColors.primary
                            : Colors.grey.shade300,
                      ),
                    ),
                ],
              ),
            );
          }),
        );
      },
    );
  }

  String _getShortLabel(OrderStatus status) {
    switch (status) {
      case OrderStatus.pending:
        return 'Ordered';
      case OrderStatus.confirmed:
        return 'Confirmed';
      case OrderStatus.preparing:
        return 'Preparing';
      case OrderStatus.outForDelivery:
        return 'On way';
      case OrderStatus.delivered:
        return 'Delivered';
      case OrderStatus.cancelled:
        return 'Cancelled';
    }
  }
}
