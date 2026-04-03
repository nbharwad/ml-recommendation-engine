const admin = require('firebase-admin');
const serviceAccount = require('../service-account.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();
const auth = admin.auth();

async function migrate() {
  console.log('Starting migration...');
  
  const snapshot = await db.collection('users').where('isAdmin', '==', true).get();
  let migrated = 0;
  
  for (const doc of snapshot.docs) {
    try {
      await auth.setCustomUserClaims(doc.id, { role: 'admin' });
      await doc.ref.update({ role: 'admin' });
      console.log(`Admin: ${doc.id}`);
      migrated++;
    } catch (error) {
      console.error(`Error ${doc.id}:`, error.message);
    }
  }
  
  console.log(`✅ Migration complete. ${migrated} admins synced.`);
  process.exit(0);
}

migrate();