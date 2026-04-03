class Validators {
  static const int maxNameLength = 100;
  static const int maxAddressLength = 500;
  static const int maxPincodeLength = 10;
  static const int maxNotesLength = 1000;

  static String? validatePhone(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter phone number';
    }
    final cleaned = value.replaceAll(RegExp(r'[^\d]'), '');
    if (cleaned.length != 10) {
      return 'Please enter 10 digit phone number';
    }
    if (!RegExp(r'^[6789]\d{9}$').hasMatch(cleaned)) {
      return 'Invalid phone number';
    }
    return null;
  }

  static String? validateOtp(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter OTP';
    }
    if (value.length != 6) {
      return 'Please enter 6 digit OTP';
    }
    if (!RegExp(r'^\d{6}$').hasMatch(value)) {
      return 'OTP must contain only numbers';
    }
    return null;
  }

  static String? validateName(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter name';
    }
    final trimmed = value.trim();
    if (trimmed.length < 2) {
      return 'Name must be at least 2 characters';
    }
    if (trimmed.length > maxNameLength) {
      return 'Name must be less than $maxNameLength characters';
    }
    if (!RegExp(r"^[a-zA-Z\s\-.]+$").hasMatch(trimmed)) {
      return 'Name can only contain letters, spaces, hyphens, and dots';
    }
    return null;
  }

  static String? validateAddress(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter address';
    }
    final trimmed = value.trim();
    if (trimmed.length < 10) {
      return 'Please enter complete address';
    }
    if (trimmed.length > maxAddressLength) {
      return 'Address must be less than $maxAddressLength characters';
    }
    return null;
  }

  static String? validatePincode(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter pincode';
    }
    if (value.length != 6) {
      return 'Please enter 6 digit pincode';
    }
    if (!RegExp(r'^\d{6}$').hasMatch(value)) {
      return 'Pincode must contain only numbers';
    }
    return null;
  }

  static String? validateRequired(String? value, String fieldName) {
    if (value == null || value.isEmpty) {
      return 'Please enter $fieldName';
    }
    final trimmed = value.trim();
    if (trimmed.length > 500) {
      return '$fieldName is too long';
    }
    return null;
  }

  static String? validateNotes(String? value) {
    if (value == null || value.isEmpty) {
      return null;
    }
    if (value.length > maxNotesLength) {
      return 'Notes must be less than $maxNotesLength characters';
    }
    return null;
  }

  static String sanitizeInput(String input) {
    return input
        .trim()
        .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), '')
        .replaceAll(RegExp(r'<[^>]*>'), '');
  }

  static String? validateQuantity(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter quantity';
    }
    final quantity = int.tryParse(value);
    if (quantity == null) {
      return 'Invalid quantity';
    }
    if (quantity < 1) {
      return 'Quantity must be at least 1';
    }
    if (quantity > 999) {
      return 'Quantity cannot exceed 999';
    }
    return null;
  }

  static String? validateSearch(String? value) {
    if (value == null || value.isEmpty) {
      return null;
    }
    if (value.length > 100) {
      return 'Search term too long';
    }
    final searchPattern = RegExp(r'[<>"\x27;\\]');
    if (searchPattern.hasMatch(value)) {
      return 'Invalid search characters';
    }
    return null;
  }
}
