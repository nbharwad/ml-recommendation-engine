import 'package:mocktail/mocktail.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';

class MockFirebaseAuth extends Mock implements FirebaseAuth {}

class MockFirebaseStorage extends Mock implements FirebaseStorage {}

class MockUser extends Mock implements User {
  @override
  String get uid => 'test-uid-123';
  
  @override
  String? get phoneNumber => '+911234567890';
  
  @override
  String? get email => 'test@example.com';
  
  @override
  bool get isAnonymous => false;
  
  @override
  bool get isEmailVerified => true;
}

class MockUserCredential extends Mock implements UserCredential {
  @override
  User? get user => MockUser();
}

class MockAuthCredential extends Mock implements AuthCredential {}

class MockFirestore extends Mock implements FirebaseFirestore {}

class MockCollectionReference extends Mock implements CollectionReference<Map<String, dynamic>> {
  @override
  String get path => 'test_collection';
}

class MockQuerySnapshot extends Mock implements QuerySnapshot<Map<String, dynamic>> {
  final List<QueryDocumentSnapshot<Map<String, dynamic>>> _docs;
  
  MockQuerySnapshot(this._docs);
  
  @override
  List<QueryDocumentSnapshot<Map<String, dynamic>>> get docs => _docs;
}

class MockQueryDocumentSnapshot extends Mock implements QueryDocumentSnapshot<Map<String, dynamic>> {
  final Map<String, dynamic> _docData;
  final String _id;
  
  MockQueryDocumentSnapshot(this._docData, this._id);
  
  @override
  Map<String, dynamic> data() => _docData;
  
  @override
  String get id => _id;
}

class MockDocumentReference extends Mock implements DocumentReference<Map<String, dynamic>> {
  @override
  String get id => 'doc-123';
  
  @override
  String get path => 'collection/doc-123';
}

class MockDocumentSnapshot extends Mock implements DocumentSnapshot<Map<String, dynamic>> {
  final Map<String, dynamic>? _docData;
  final String _id;
  final bool _exists;
  
  MockDocumentSnapshot(this._docData, this._id, {bool exists = true}) : _exists = exists;
  
  @override
  Map<String, dynamic>? data() => _docData;
  
  @override
  String get id => _id;
  
  @override
  bool get exists => _exists;
}

class MockStorageReference extends Mock implements Reference {
  @override
  String get name => 'test-file.jpg';
  
  @override
  String get bucket => 'test-bucket.appspot.com';
  
  @override
  String get fullPath => 'prescriptions/test-file.jpg';
}

class MockUploadTask extends Mock implements UploadTask {
  @override
  TaskSnapshot get snapshot => MockTaskSnapshot();
}

class MockTaskSnapshot extends Mock implements TaskSnapshot {
  @override
  Reference get ref => MockStorageReference();
  
  @override
  int get totalBytes => 1000;
  
  @override
  TaskState get state => TaskState.success;
}

void registerFallbackValues() {
  registerFallbackValue(MockAuthCredential());
  registerFallbackValue(<String, dynamic>{});
  registerFallbackValue(any());
}
