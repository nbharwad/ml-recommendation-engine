import 'dart:math' as math;

import 'package:flutter/material.dart';

extension ResponsiveContext on BuildContext {
  Size get screenSize => MediaQuery.sizeOf(this);
  double get screenWidth => screenSize.width;
  double get screenHeight => screenSize.height;

  bool get isCompactWidth => screenWidth < 360;
  bool get isMediumWidth => screenWidth >= 360 && screenWidth < 600;
  bool get isShortHeight => screenHeight < 700;

  double adaptiveSpace(double base) {
    if (isCompactWidth) {
      return math.max(8, base * 0.72);
    }
    if (isMediumWidth) {
      return base * 0.88;
    }
    return base;
  }

  EdgeInsets get pagePadding {
    final horizontal = screenWidth >= 600 ? 24.0 : (isCompactWidth ? 16.0 : 20.0);
    return EdgeInsets.symmetric(horizontal: horizontal, vertical: adaptiveSpace(20));
  }

  double get gridAspectRatio {
    if (screenWidth < 380) return 0.60;
    if (screenWidth < 600) return 0.68;
    return 0.72;
  }

  int get gridCrossAxisCount {
    if (screenWidth >= 960) return 4;
    if (screenWidth >= 700) return 3;
    return 2;
  }

  double adaptiveFontSize(double base) {
    if (isCompactWidth) {
      return math.max(10.0, base * 0.85);
    }
    return base;
  }
}
