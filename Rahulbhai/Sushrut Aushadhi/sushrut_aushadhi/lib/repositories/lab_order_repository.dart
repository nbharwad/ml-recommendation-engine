import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/lab_order_model.dart';
import '../core/di/service_providers.dart';
import '../services/lab_service.dart';

class LabOrderRepository {
  final LabService _labService;

  LabOrderRepository(this._labService);

  static Provider<LabOrderRepository> get provider => Provider<LabOrderRepository>((ref) {
    return LabOrderRepository(ref.read(labServiceProvider));
  });

  String generateLabOrderId() => _labService.generateLabOrderId();

  Future<Map<String, dynamic>> getTodayLabSummary() async {
    return _labService.getTodayLabSummary();
  }

  Future<String> placeLabOrder(LabOrderModel order) async {
    return _labService.placeLabOrder(order);
  }

  Stream<List<LabOrderModel>> getUserLabOrders(String userId) {
    return _labService.getUserLabOrders(userId);
  }

  Stream<List<LabOrderModel>> getAllLabOrders() {
    return _labService.getAllLabOrders();
  }

  Future<LabOrderModel?> getLabOrderById(String orderId) async {
    return _labService.getLabOrderById(orderId);
  }

  Stream<LabOrderModel?> getLabOrderStream(String orderId) {
    return _labService.getLabOrderStream(orderId);
  }

  Future<void> updateLabOrderStatus(String orderId, LabOrderStatus status, {String? note, String? updatedBy}) async {
    return _labService.updateLabOrderStatus(orderId, status, note: note, updatedBy: updatedBy);
  }

  Future<void> updatePaymentStatus(String orderId, String paymentStatus) async {
    return _labService.updatePaymentStatus(orderId, paymentStatus);
  }

  Future<String> uploadLabResult(String orderId, String filePath, String fileName) async {
    return _labService.uploadLabResult(orderId, filePath, fileName);
  }
}