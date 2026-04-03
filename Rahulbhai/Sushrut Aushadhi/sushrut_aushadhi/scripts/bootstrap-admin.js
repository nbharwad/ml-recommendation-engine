const admin = require('firebase-admin');
const serviceAccount = require('../service-account.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();
const auth = admin.auth();

async function bootstrap() {
  const uid = process.argv[2];
  
  if (!uid) {
    console.log('Usage: node bootstrap-admin.js <user-uid>');
    console.log('\nTo get your UID:');
    console.log('1. Go to Firebase Console → Authentication → Users');
    console.log('2. Copy your user UID');
    process.exit(1);
  }
  
  try {
    await auth.setCustomUserClaims(uid, { role: 'admin' });
    await db.collection('users').doc(uid).update({ 
      isAdmin: true,
      role: 'admin'
    });
    console.log(`✅ Successfully set user ${uid} as admin`);
  } catch (error) {
    console.error('Error:', error.message);
  }
  
  process.exit(0);
}

bootstrap();