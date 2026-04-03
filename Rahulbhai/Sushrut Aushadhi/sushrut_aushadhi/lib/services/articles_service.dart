import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/article_model.dart';

class ArticlesService {
  static const String _apiKey = '8108034f3afb46848071326f1a77dfd7';

  static Future<List<ArticleModel>> getHealthArticles() async {
    try {
      final url = Uri.parse(
        'https://newsapi.org/v2/everything?'
        'q=health+medicine+wellness+india&'
        'language=en&'
        'sortBy=publishedAt&'
        'pageSize=10&'
        'apiKey=$_apiKey'
      );

      final response = await http.get(url).timeout(
        const Duration(seconds: 10),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final articles = data['articles'] as List;
        return articles
            .where((a) => a['title'] != null && a['title'] != '[Removed]')
            .map((a) => ArticleModel.fromJson(a))
            .toList();
      }
    } catch (e) {
      print('Articles fetch error: $e');
    }

    return _getFallbackArticles();
  }

  static List<ArticleModel> _getFallbackArticles() {
    return [
      ArticleModel(
        title: 'Benefits of Vitamin D for Immunity',
        description: 'Vitamin D plays a crucial role in immune function. Here is how to maintain optimal levels.',
        url: 'https://www.healthline.com',
        imageUrl: '',
        source: 'Healthline',
        publishedAt: DateTime.now(),
      ),
      ArticleModel(
        title: 'Stay Hydrated: Why Water Matters',
        description: 'Drinking enough water daily supports medicine absorption and overall health.',
        url: 'https://www.webmd.com',
        imageUrl: '',
        source: 'WebMD',
        publishedAt: DateTime.now(),
      ),
      ArticleModel(
        title: 'Managing Diabetes with Diet & Medicine',
        description: 'A combination of healthy diet and prescribed medicines helps manage diabetes effectively.',
        url: 'https://www.diabetes.org',
        imageUrl: '',
        source: 'Diabetes.org',
        publishedAt: DateTime.now(),
      ),
    ];
  }
}
