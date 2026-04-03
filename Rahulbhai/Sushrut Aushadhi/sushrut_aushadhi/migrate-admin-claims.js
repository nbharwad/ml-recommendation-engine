const admin = require('firebase-admin');

const serviceAccount = require('./service-account.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

async function migrateAdminClaims() {
  console.log('Starting admin claims migration...');
  
  try {
    const snapshot = await db.collection('users').where('isAdmin', '==', true).get();
    
    console.log(`Found ${snapshot.size} users with isAdmin: true`);
    
    let successCount = 0;
    let errorCount = 0;
    const errors = [];
    
    for (const doc of snapshot.docs) {
      const uid = doc.id;
      const userData = doc.data();
      
      try {
        await admin.auth().setCustomUserClaims(uid, { role: 'admin' });
        console.log(`✓ Set custom claims for: ${uid} (${userData.name || 'unnamed'})`);
        successCount++;
      } catch (error) {
        console.error(`✗ Error setting claims for ${uid}:`, error.message);
        errors.push({ uid, error: error.message });
        errorCount++;
      }
    }
    
    console.log('\n--- Migration Summary ---');
    console.log(`Total admins found: ${snapshot.size}`);
    console.log(`Successfully migrated: ${successCount}`);
    console.log(`Errors: ${errorCount}`);
    
    if (errors.length > 0) {
      console.log('\nErrors:');
      errors.forEach(e => console.log(`  - ${e.uid}: ${e.error}`));
    }
    
    console.log('\nMigration complete! Custom claims will take effect after user token refresh.');
    
  } catch (error) {
    console.error('Migration failed:', error);
  } finally {
    process.exit();
  }
}

migrateAdminClaims();
