import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class SearchHistoryService {
  static const String _key = 'search_history';
  static const int _maxItems = 5;

  static Future<List<String>> getHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_key);
    if (data == null) return [];
    return List<String>.from(jsonDecode(data));
  }

  static Future<void> addSearch(String query) async {
    if (query.trim().isEmpty) return;
    final query0 = query.trim();

    final prefs = await SharedPreferences.getInstance();
    final history = await getHistory();

    history.remove(query0);
    history.insert(0, query0);

    final trimmed = history.take(_maxItems).toList();

    await prefs.setString(_key, jsonEncode(trimmed));
  }

  static Future<void> removeSearch(String query) async {
    final prefs = await SharedPreferences.getInstance();
    final history = await getHistory();
    history.remove(query);
    await prefs.setString(_key, jsonEncode(history));
  }

  static Future<void> clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
