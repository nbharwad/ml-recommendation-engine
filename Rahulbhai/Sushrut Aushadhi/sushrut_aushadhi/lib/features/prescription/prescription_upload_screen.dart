import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/widgets/login_prompt_widget.dart';
import '../../services/connectivity_service.dart';
import '../../services/prescription_service.dart';
import '../../providers/notification_provider.dart';
import '../../core/constants/app_colors.dart';

class PrescriptionUploadScreen
    extends ConsumerStatefulWidget {
  const PrescriptionUploadScreen({super.key});
  ConsumerState<PrescriptionUploadScreen> createState() =>
      _PrescriptionUploadScreenState();
}

class _PrescriptionUploadScreenState
    extends ConsumerState<PrescriptionUploadScreen> {

  final _service = PrescriptionService();
  File? _selectedImage;
  bool _isUploading = false;
  double _uploadProgress = 0.0;
  String? _errorMessage;

  // ── CHECK LOGIN ────────────────────────────────────────
  bool get _isLoggedIn =>
      FirebaseAuth.instance.currentUser != null;

  // ── PICK IMAGE ─────────────────────────────────────────
  Future<void> _pickImage(ImageSource source) async {
    setState(() => _errorMessage = null);

    try {
      File? file;

      if (source == ImageSource.camera) {
        file = await _service.pickFromCamera();
      } else {
        file = await _service.pickFromGallery();
      }

      if (file != null) {
        setState(() => _selectedImage = file);
      }

    } on PermissionException catch (e) {
      setState(() => _errorMessage = e.message);
      _showPermissionDialog(source);

    } catch (e) {
      setState(() =>
          _errorMessage = e.toString()
              .replaceAll('Exception: ', ''));
    }
  }

  // ── SHOW PERMISSION DENIED DIALOG ─────────────────────
  void _showPermissionDialog(ImageSource source) {
    final isCamera = source == ImageSource.camera;
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16)),
        title: Text(
            isCamera
                ? 'Camera Permission Required'
                : 'Storage Permission Required',
            style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700)),
        content: Text(
          isCamera
              ? 'Please allow camera access to '
                'take prescription photos. '
                'Go to Settings → Apps → '
                'Sushrut Aushadhi → Permissions → '
                'Camera → Allow.'
              : 'Please allow storage access to '
                'upload prescription photos. '
                'Go to Settings → Apps → '
                'Sushrut Aushadhi → Permissions → '
                'Storage → Allow.',
          style: TextStyle(fontSize: 13,
              color: Colors.grey[700]),
        ),
        actions: [
          TextButton(
            onPressed: () =>
                Navigator.pop(context),
            child: Text('Cancel',
                style: TextStyle(
                    color: Colors.grey))),
          ElevatedButton(
            onPressed: () async {
              Navigator.pop(context);
              await openAppSettings();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor:
                  AppColors.primary,
              shape: RoundedRectangleBorder(
                  borderRadius:
                      BorderRadius.circular(8)),
              elevation: 0,
            ),
            child: Text('Open Settings',
                style: TextStyle(
                    color: Colors.white))),
        ],
      ),
    );
  }

  // ── SHOW IMAGE SOURCE PICKER ───────────────────────────
  void _showImageSourcePicker() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
            top: Radius.circular(20))),
      builder: (_) => SafeArea(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [

              // Handle bar
              Container(
                width: 40, height: 4,
                margin: EdgeInsets.only(bottom: 20),
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius:
                      BorderRadius.circular(2)),
              ),

              Text('Select Photo Source',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF1A1A1A),
                )),

              SizedBox(height: 20),

              // Camera option
              _sourceOption(
                icon: Icons.camera_alt_outlined,
                title: 'Take Photo',
                subtitle:
                    'Use camera to capture prescription',
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),

              SizedBox(height: 12),

              // Gallery option
              _sourceOption(
                icon: Icons.photo_library_outlined,
                title: 'Choose from Gallery',
                subtitle:
                    'Select existing photo',
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),

              SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sourceOption({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Color(0xFFF7F9F7),
          borderRadius:
              BorderRadius.circular(12),
          border: Border.all(
              color: Color(0xFFEDF2ED)),
        ),
        child: Row(
          children: [
            Container(
              width: 44, height: 44,
              decoration: BoxDecoration(
                color: Color(0xFFE1F5EE),
                borderRadius:
                    BorderRadius.circular(10),
              ),
              child: Icon(icon,
                  color: AppColors.primary,
                  size: 22),
            ),
            SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  Text(title,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1A1A1A),
                    )),
                  Text(subtitle,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey,
                    )),
                ],
              ),
            ),
            Icon(Icons.arrow_forward_ios,
                size: 14, color: Colors.grey),
          ],
        ),
      ),
    );
  }

  // ── UPLOAD PRESCRIPTION ────────────────────────────────
  Future<void> _uploadPrescription() async {
    if (_selectedImage == null) return;

    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      context.push('/login');
      return;
    }

    // Check internet
    final isOnline =
        await ConnectivityService.checkConnection();
    if (!isOnline) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
              'No internet. Please connect '
              'to upload.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() {
      _isUploading = true;
      _uploadProgress = 0.0;
      _errorMessage = null;
    });

    try {
      // Get user details
      String userName = 'User';
      String userPhone = '';
      try {
        final userDoc = await FirebaseFirestore
            .instance
            .collection('users')
            .doc(user.uid)
            .get();
        if (userDoc.exists) {
          userName = userDoc.data()?['name']
              ?? user.displayName
              ?? 'User';
          userPhone =
              userDoc.data()?['phone'] ?? '';
        }
      } catch (e) {
        userName = user.displayName ?? 'User';
      }

      // Upload image
      final url =
          await _service.uploadPrescription(
        userId: user.uid,
        imageFile: _selectedImage!,
        onProgress: (p) {
          if (mounted) {
            setState(() =>
                _uploadProgress = p);
          }
        },
      );

      // Save to Firestore
      await _service.savePrescription(
        userId: user.uid,
        userPhone: userPhone,
        userName: userName,
        imageUrl: url,
      );

      // Add notification
      await ref
          .read(notificationProvider.notifier)
          .addNotification(
            title: 'Prescription Uploaded',
            body: 'Your prescription is under '
                'review by our pharmacist.',
            type: 'prescription',
          );

      if (!mounted) return;
      setState(() => _isUploading = false);
      _showSuccessDialog();

    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isUploading = false;
        _errorMessage = e.toString()
            .replaceAll('Exception: ', '');
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
              'Upload failed: $_errorMessage'),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 4),
        ),
      );
    }
  }

  // ── SUCCESS DIALOG ─────────────────────────────────────
  void _showSuccessDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(
            borderRadius:
                BorderRadius.circular(16)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(height: 8),
            Container(
              width: 70, height: 70,
              decoration: BoxDecoration(
                color: Color(0xFFE1F5EE),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.check_circle,
                  color: AppColors.primary,
                  size: 40)),
            SizedBox(height: 16),
            Text('Prescription Uploaded!',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
              )),
            SizedBox(height: 8),
            Text(
              'Our pharmacist will review it '
              'shortly. You can view it in '
              'My Prescriptions.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 13,
                color: Colors.grey,
                height: 1.5,
              )),
          ],
        ),
        actions: [
          SizedBox(
            width: double.infinity,
            child: Column(
              children: [
                ElevatedButton(
                  onPressed: () {
                    Navigator.pop(context);
                    context.push(
                        '/my-prescriptions');
                  },
                  style: ElevatedButton
                      .styleFrom(
                    backgroundColor:
                        AppColors.primary,
                    minimumSize:
                        Size(double.infinity,
                            44),
                    shape:
                        RoundedRectangleBorder(
                      borderRadius:
                          BorderRadius
                              .circular(10)),
                    elevation: 0,
                  ),
                  child: Text(
                    'View My Prescriptions',
                    style: TextStyle(
                        color: Colors.white,
                        fontWeight:
                            FontWeight.w600),
                  ),
                ),
                SizedBox(height: 8),
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    context.go('/cart');
                  },
                  child: Text('Go to Cart',
                    style: TextStyle(
                      color: AppColors.primary,
                      fontWeight:
                          FontWeight.w600,
                    )),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── BUILD ──────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    // Show login prompt if not logged in
    if (!_isLoggedIn) {
      return Scaffold(
        backgroundColor: Color(0xFFF7F9F7),
        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 0,
          title: Text('Upload Prescription',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1A1A1A),
            )),
          leading: IconButton(
            icon: Icon(Icons.arrow_back,
                color: Color(0xFF1A1A1A)),
            onPressed: () =>
                context.pop()),
        ),
        body: LoginPromptWidget(
          message: 'Please login to upload '
              'and manage your prescriptions.',
        ),
      );
    }

    return Scaffold(
      backgroundColor: Color(0xFFF7F9F7),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text('Upload Prescription',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: Color(0xFF1A1A1A),
          )),
        leading: IconButton(
          icon: Icon(Icons.arrow_back,
              color: Color(0xFF1A1A1A)),
          onPressed: () => context.pop()),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.start,
            children: [

              // ── PHARMACIST TRUST BADGE ─────
              Container(
                padding: EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Color(0xFFE1F5EE),
                  borderRadius:
                      BorderRadius.circular(12),
                  border: Border.all(
                      color:
                          Color(0xFF9FE1CB)),
                ),
                child: Row(
                  children: [
                    Icon(Icons.verified_user,
                      size: 28,
                      color: AppColors.primary),
                    SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment:
                            CrossAxisAlignment
                                .start,
                        children: [
                          Text(
                            'Verified by Pharmacist',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight:
                                  FontWeight.w700,
                              color:
                                AppColors.primary,
                            )),
                          Text(
                            'Your prescription will be '
                            'reviewed before dispensing.',
                            style: TextStyle(
                              fontSize: 11,
                              color:
                                Color(0xFF1D9E75),
                              height: 1.5,
                            )),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              SizedBox(height: 16),

              // ── IMAGE PREVIEW BOX ──────────
              GestureDetector(
                onTap: _showImageSourcePicker,
                child: Container(
                  width: double.infinity,
                  height: 200,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius:
                        BorderRadius.circular(16),
                    border: Border.all(
                      color: _selectedImage != null
                          ? AppColors.primary
                          : Color(0xFFEDF2ED),
                      width: _selectedImage != null
                          ? 2 : 1,
                      style: _selectedImage != null
                          ? BorderStyle.solid
                          : BorderStyle.solid,
                    ),
                  ),
                  child: _selectedImage != null
                      ? ClipRRect(
                          borderRadius:
                              BorderRadius
                                  .circular(14),
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              Image.file(
                                _selectedImage!,
                                fit: BoxFit.cover,
                              ),
                              Positioned(
                                top: 8,
                                right: 8,
                                child: GestureDetector(
                                  onTap: () =>
                                    setState(() =>
                                      _selectedImage =
                                        null),
                                  child: Container(
                                    padding:
                                      EdgeInsets
                                        .all(6),
                                    decoration:
                                      BoxDecoration(
                                        color: Colors
                                          .black54,
                                        shape: BoxShape
                                          .circle,
                                      ),
                                    child: Icon(
                                      Icons.close,
                                      color:
                                        Colors.white,
                                      size: 16),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        )
                      : Column(
                          mainAxisAlignment:
                              MainAxisAlignment
                                  .center,
                          children: [
                            Container(
                              width: 60,
                              height: 60,
                              decoration: BoxDecoration(
                                color:
                                  Color(0xFFE1F5EE),
                                shape:
                                  BoxShape.circle,
                              ),
                              child: Icon(
                                Icons
                                  .add_photo_alternate_outlined,
                                color:
                                  AppColors.primary,
                                size: 28),
                            ),
                            SizedBox(height: 12),
                            Text(
                              'Tap to add prescription',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight:
                                  FontWeight.w600,
                                color:
                                  Color(0xFF1A1A1A),
                              )),
                            SizedBox(height: 4),
                            Text(
                              'Camera or Gallery',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey,
                              )),
                          ],
                        ),
                ),
              ),

              SizedBox(height: 12),

              // ── ACTION BUTTONS ─────────────
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () =>
                          _pickImage(
                              ImageSource.camera),
                      icon: Icon(
                        Icons.camera_alt_outlined,
                        size: 18,
                        color: AppColors.primary),
                      label: Text('Camera',
                        style: TextStyle(
                          color: AppColors.primary,
                          fontWeight:
                            FontWeight.w600,
                        )),
                      style: OutlinedButton
                          .styleFrom(
                        padding: EdgeInsets
                            .symmetric(
                              vertical: 12),
                        side: BorderSide(
                          color:
                            AppColors.primary),
                        shape:
                          RoundedRectangleBorder(
                            borderRadius:
                              BorderRadius
                                .circular(10)),
                      ),
                    ),
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () =>
                          _pickImage(
                              ImageSource.gallery),
                      icon: Icon(
                        Icons
                          .photo_library_outlined,
                        size: 18,
                        color: AppColors.primary),
                      label: Text('Gallery',
                        style: TextStyle(
                          color: AppColors.primary,
                          fontWeight:
                            FontWeight.w600,
                        )),
                      style: OutlinedButton
                          .styleFrom(
                        padding: EdgeInsets
                            .symmetric(
                              vertical: 12),
                        side: BorderSide(
                          color:
                            AppColors.primary),
                        shape:
                          RoundedRectangleBorder(
                            borderRadius:
                              BorderRadius
                                .circular(10)),
                      ),
                    ),
                  ),
                ],
              ),

              SizedBox(height: 16),

              // ── ERROR MESSAGE ──────────────
              if (_errorMessage != null)
                Container(
                  padding: EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Color(0xFFFFEBEE),
                    borderRadius:
                        BorderRadius.circular(10),
                    border: Border.all(
                        color:
                            Color(0xFFF7C1C1)),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.error_outline,
                          color: Colors.red,
                          size: 18),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _errorMessage!,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.red[700],
                          )),
                      ),
                    ],
                  ),
                ),

              if (_errorMessage != null)
                SizedBox(height: 12),

              // ── UPLOAD PROGRESS ────────────
              if (_isUploading) ...[
                ClipRRect(
                  borderRadius:
                      BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: _uploadProgress,
                    backgroundColor:
                        Color(0xFFE1F5EE),
                    valueColor:
                        AlwaysStoppedAnimation(
                            AppColors.primary),
                    minHeight: 6,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  'Uploading... '
                  '${(_uploadProgress * 100).round()}%',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12,
                    color: AppColors.primary,
                    fontWeight: FontWeight.w600,
                  )),
                SizedBox(height: 16),
              ],

              // ── TIPS CARD ──────────────────
              Container(
                padding: EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius:
                      BorderRadius.circular(12),
                  border: Border.all(
                      color: Color(0xFFEDF2ED)),
                ),
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Tips for a good photo',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF1A1A1A),
                      )),
                    SizedBox(height: 10),
                    _tip('Good lighting, '
                        'no shadows'),
                    _tip('All text clearly '
                        'visible'),
                    _tip('Doctor\'s stamp '
                        'included'),
                    _tip('All 4 corners '
                        'visible'),
                  ],
                ),
              ),

              SizedBox(height: 16),

              // ── LEGAL DISCLAIMER ───────────
              Container(
                padding: EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Color(0xFFFFF8E1),
                  borderRadius:
                      BorderRadius.circular(10),
                  border: Border.all(
                      color: Color(0xFFFFE082)),
                ),
                child: Row(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.warning_amber_rounded,
                      size: 14,
                      color: Color(0xFFE65100)),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'By uploading, you confirm '
                        'this is a valid prescription '
                        'issued by a registered medical '
                        'practitioner as per Indian '
                        'drug laws.',
                        style: TextStyle(
                          fontSize: 11,
                          color: Color(0xFFE65100),
                          height: 1.5,
                        )),
                    ),
                  ],
                ),
              ),

              SizedBox(height: 20),

              // ── UPLOAD BUTTON ──────────────
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _selectedImage == null
                      || _isUploading
                      ? null
                      : _uploadPrescription,
                  style: ElevatedButton.styleFrom(
                    backgroundColor:
                        AppColors.primary,
                    disabledBackgroundColor:
                        Color(0xFFB4B2A9),
                    padding: EdgeInsets.symmetric(
                        vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius:
                          BorderRadius.circular(12)),
                    elevation: 0,
                  ),
                  child: _isUploading
                      ? Row(
                          mainAxisAlignment:
                              MainAxisAlignment
                                  .center,
                          children: [
                            SizedBox(
                              width: 18,
                              height: 18,
                              child:
                                CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                )),
                            SizedBox(width: 10),
                            Text('Uploading...',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight:
                                  FontWeight.w700,
                                color: Colors.white,
                              )),
                          ],
                        )
                      : Text(
                          _selectedImage != null
                              ? 'Upload Prescription'
                              : 'Select Photo First',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          )),
                ),
              ),

              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _tip(String text) {
    return Padding(
      padding: EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          Icon(Icons.check_circle,
              size: 14,
              color: AppColors.primary),
          SizedBox(width: 8),
          Expanded(
            child: Text(text,
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[700],
              )),
          ),
        ],
      ),
    );
  }
}