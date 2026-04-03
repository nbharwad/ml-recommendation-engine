import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/article_model.dart';
import '../services/articles_service.dart';

final articlesProvider = FutureProvider<List<ArticleModel>>((ref) {
  return ArticlesService.getHealthArticles();
});
