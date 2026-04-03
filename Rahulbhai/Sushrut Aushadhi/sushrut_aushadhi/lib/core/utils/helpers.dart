import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../models/order_model.dart';
import '../constants/app_colors.dart';

class Helpers {
  static String formatPrice(double price) {
    return 'Rs ${price.toStringAsFixed(2)}';
  }

  static String formatDate(DateTime date) {
    return DateFormat('dd MMM yyyy').format(date);
  }

  static String formatDateTime(DateTime date) {
    return DateFormat('dd MMM yyyy, hh:mm a').format(date);
  }

  static String formatOrderId(String orderId) {
    if (orderId.length < 8) {
      return '#${orderId.toUpperCase()}';
    }
    return '#${orderId.substring(0, 8).toUpperCase()}';
  }

  static Color getStatusColor(OrderStatus status) {
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

  static String getCategoryDisplayName(String category) {
    switch (category) {
      case 'all':
        return 'All';
      case 'pain_relief':
        return 'Pain Relief';
      case 'fever':
        return 'Fever';
      case 'vitamins':
        return 'Vitamins';
      case 'antibiotics':
        return 'Antibiotics';
      default:
        return 'Other';
    }
  }

  static String getCategoryIcon(String category) {
    switch (category) {
      case 'all':
        return 'All';
      case 'pain_relief':
        return 'PR';
      case 'fever':
        return 'FV';
      case 'vitamins':
        return 'VT';
      case 'antibiotics':
        return 'AB';
      default:
        return 'OT';
    }
  }

  static int calculateDiscount(double price, double mrp) {
    if (mrp <= 0) return 0;
    return ((mrp - price) / mrp * 100).round();
  }
}
