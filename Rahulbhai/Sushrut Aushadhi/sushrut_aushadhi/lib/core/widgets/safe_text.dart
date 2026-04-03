import 'package:flutter/material.dart';

class SafeText extends StatelessWidget {
  final String text;
  final TextStyle? style;
  final int? maxLines;
  final TextOverflow overflow;
  final TextAlign? textAlign;
  final TextDirection? textDirection;

  const SafeText(
    this.text, {
    super.key,
    this.style,
    this.maxLines,
    this.overflow = TextOverflow.ellipsis,
    this.textAlign,
    this.textDirection,
  });

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: style,
      maxLines: maxLines,
      overflow: overflow,
      textAlign: textAlign,
      textDirection: textDirection,
    );
  }
}

class SafePriceText extends StatelessWidget {
  final String price;
  final TextStyle? style;
  final double? fontSize;

  const SafePriceText(
    this.price, {
    super.key,
    this.style,
    this.fontSize,
  });

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.scaleDown,
      alignment: Alignment.centerLeft,
      child: Text(
        price,
        style: style ??
            TextStyle(
              fontSize: fontSize ?? 14,
              fontWeight: FontWeight.bold,
            ),
      ),
    );
  }
}