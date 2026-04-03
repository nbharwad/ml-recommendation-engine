import 'package:url_launcher/url_launcher.dart';

class WhatsAppService {
  static Future<void> sendOrderUpdate({
    required String customerPhone,
    required String orderId,
    required String customerName,
    required String status,
    required double totalAmount,
    required List<String> itemNames,
    required String storePhone,
  }) async {
    final message = _buildMessage(
      orderId: orderId,
      customerName: customerName,
      status: status,
      totalAmount: totalAmount,
      itemNames: itemNames,
      storePhone: storePhone,
    );

    String phone = customerPhone
        .replaceAll('+', '')
        .replaceAll(' ', '')
        .replaceAll('-', '');
    if (phone.startsWith('0')) phone = '91${phone.substring(1)}';
    if (phone.length == 10) phone = '91$phone';

    final encoded = Uri.encodeComponent(message);
    final url = 'https://wa.me/$phone?text=$encoded';

    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } else {
      throw Exception('WhatsApp not installed on this device');
    }
  }

  static String _buildMessage({
    required String orderId,
    required String customerName,
    required String status,
    required double totalAmount,
    required List<String> itemNames,
    required String storePhone,
  }) {
    final shortId = orderId.length > 6
        ? orderId.substring(orderId.length - 6).toUpperCase()
        : orderId.toUpperCase();

    String itemsText = '';
    if (itemNames.length <= 3) {
      itemsText = itemNames.map((i) => '  • $i').join('\n');
    } else {
      final shown = itemNames.take(3).map((i) => '  • $i').join('\n');
      itemsText = '$shown\n  • +${itemNames.length - 3} more items';
    }

    switch (status) {
      case 'confirmed':
        return '''✅ *Order Confirmed!*

Namaste $customerName 🙏

Your order at *Sushrut Aushadhi* has been confirmed.

🧾 *Order #SA-$shortId*
💊 *Items:*
$itemsText

💰 *Total: ₹${totalAmount.toStringAsFixed(0)}*
💵 Payment: Cash on Delivery

Our pharmacist is preparing your order. We will notify you when it is out for delivery.

📞 Questions? Call: $storePhone

_Thank you for choosing Sushrut Aushadhi_ 🌿''';

      case 'preparing':
        return '''🔄 *Order Being Prepared*

Namaste $customerName 🙏

Good news! Your medicines are being packed at *Sushrut Aushadhi*.

🧾 *Order #SA-$shortId*
💊 *Items:*
$itemsText

💰 *Total: ₹${totalAmount.toStringAsFixed(0)}*

We will send another message when your order is out for delivery.

📞 Questions? Call: $storePhone

_Sushrut Aushadhi_ 🌿''';

      case 'out_for_delivery':
        return '''🚚 *Out for Delivery!*

Namaste $customerName 🙏

Your order from *Sushrut Aushadhi* is on its way! 🏍️

🧾 *Order #SA-$shortId*
💊 *Items:*
$itemsText

💰 *Total: ₹${totalAmount.toStringAsFixed(0)}*
💵 Please keep *cash ready* for payment

Expected delivery: Within 1-2 hours

📞 Delivery queries? Call: $storePhone

_Sushrut Aushadhi_ 🌿''';

      case 'delivered':
        return '''🎉 *Order Delivered!*

Namaste $customerName 🙏

Your order from *Sushrut Aushadhi* has been delivered successfully! ✓

🧾 *Order #SA-$shortId*
💊 *Items:*
$itemsText

💰 *Total Paid: ₹${totalAmount.toStringAsFixed(0)}*

We hope you feel better soon! 🌿

⭐ *Loved our service?* 
Share Sushrut Aushadhi with your family and friends!

📞 Need anything? Call: $storePhone

_Thank you for choosing Sushrut Aushadhi_ 🙏''';

      case 'cancelled':
        return '''❌ *Order Cancelled*

Namaste $customerName 🙏

Your order at *Sushrut Aushadhi* has been cancelled.

🧾 *Order #SA-$shortId*
💰 *Amount: ₹${totalAmount.toStringAsFixed(0)}*

If you have any questions or want to reorder, please call us.

📞 Call: $storePhone

_We are sorry for the inconvenience_ 🙏
_Sushrut Aushadhi_ 🌿''';

      default:
        return '''📋 *Order Update*

Namaste $customerName 🙏

Your order *#SA-$shortId* at Sushrut Aushadhi has been updated.

Status: *${status.toUpperCase()}*
💰 Total: ₹${totalAmount.toStringAsFixed(0)}

📞 Questions? Call: $storePhone

_Sushrut Aushadhi_ 🌿''';
    }
  }
}
