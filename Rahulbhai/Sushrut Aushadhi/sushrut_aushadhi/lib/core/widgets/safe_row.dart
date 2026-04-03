import 'package:flutter/material.dart';

class SafeRow extends StatelessWidget {
  final List<Widget> children;
  final MainAxisAlignment mainAxisAlignment;
  final CrossAxisAlignment crossAxisAlignment;
  final MainAxisSize mainAxisSize;
  final double? spacing;

  const SafeRow({
    super.key,
    required this.children,
    this.mainAxisAlignment = MainAxisAlignment.start,
    this.crossAxisAlignment = CrossAxisAlignment.center,
    this.mainAxisSize = MainAxisSize.max,
    this.spacing,
  });

  @override
  Widget build(BuildContext context) {
    if (children.isEmpty) return const SizedBox.shrink();
    
    if (spacing != null && spacing! > 0) {
      final wrappedChildren = <Widget>[];
      for (int i = 0; i < children.length; i++) {
        if (children[i] is Flexible || children[i] is Spacer) {
          wrappedChildren.add(children[i]);
        } else {
          wrappedChildren.add(Flexible(child: children[i]));
        }
        if (i < children.length - 1) {
          wrappedChildren.add(SizedBox(width: spacing));
        }
      }
      return Row(
        mainAxisAlignment: mainAxisAlignment,
        crossAxisAlignment: crossAxisAlignment,
        mainAxisSize: mainAxisSize,
        children: wrappedChildren,
      );
    }
    
    final wrappedChildren = children.map((child) {
      if (child is Flexible || child is Spacer || child is SizedBox) {
        return child;
      }
      return Flexible(child: child);
    }).toList();
    
    return Row(
      mainAxisAlignment: mainAxisAlignment,
      crossAxisAlignment: crossAxisAlignment,
      mainAxisSize: mainAxisSize,
      children: wrappedChildren,
    );
  }
}