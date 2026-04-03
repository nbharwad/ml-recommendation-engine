import 'package:flutter/material.dart';

class DataMasking {
  static String maskPhone(String phone) {
    if (phone.length < 4) return '****';
    final cleaned = phone.replaceAll(RegExp(r'[\s\-\+]'), '');
    final last4 = cleaned.substring(cleaned.length - 4);
    return '******$last4';
  }

  static String maskEmail(String email) {
    final parts = email.split('@');
    if (parts.length != 2) return '****';
    final name = parts[0];
    final domain = parts[1];
    final visible = name.length > 2 ? name.substring(0, 2) : name[0];
    return '$visible***@$domain';
  }

  static String maskAddress(String address) {
    if (address.isEmpty) return '****';
    final parts = address.split(',');
    if (parts.length >= 2) {
      return parts
          .sublist(parts.length >= 2 ? parts.length - 2 : 0)
          .join(',')
          .trim();
    }
    return address.length > 20 ? '${address.substring(0, 20)}...' : address;
  }

  static void showFullDataDialog(
    BuildContext context,
    String title,
    String fullData,
  ) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        title: Text(
          title,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
        ),
        content: SelectableText(
          fullData,
          style: const TextStyle(fontSize: 14),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text(
              'Close',
              style: TextStyle(color: Color(0xFF0F6E56)),
            ),
          ),
        ],
      ),
    );
  }
}