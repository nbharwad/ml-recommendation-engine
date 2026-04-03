const admin = require('firebase-admin');
const serviceAccount = require('../service-account.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const remoteConfig = admin.remoteConfig();

const parameters = {
  store_phone: { defaultValue: { value: '919429709499' } },
  store_whatsapp: { defaultValue: { value: '919429709499' } },
  store_email: { defaultValue: { value: 'contact@sushrutaushadhi.com' } },
  store_address: { defaultValue: { value: 'Shop No. X, Your Area, Bengaluru - 560XXX' } },
  store_name: { defaultValue: { value: 'Sushrut Aushadhi' } },
  drug_license_no: { defaultValue: { value: 'DL-KA-XXXXXXXX' } },
  gst_number: { defaultValue: { value: 'XXXXXXXXXXXX' } },
  news_api_key: { defaultValue: { value: '' } },
  algolia_app_id: { defaultValue: { value: '' } },
  algolia_api_key: { defaultValue: { value: '' } },
  delivery_hours: { defaultValue: { value: '2' } },
  min_order_amount: { defaultValue: { value: '0' } },
  free_delivery_above: { defaultValue: { value: '200' } },
  first_order_discount: { defaultValue: { value: '20' } },
  promo_code: { defaultValue: { value: 'SUSHRUT20' } },
  store_open_time: { defaultValue: { value: '09:00' } },
  store_close_time: { defaultValue: { value: '21:00' } },
  is_store_open: { defaultValue: { value: 'true' } },
  maintenance_mode: { defaultValue: { value: 'false' } },
  maintenance_message: { defaultValue: { value: 'App is under maintenance. We will be back soon!' } },
};

async function setupRemoteConfig() {
  try {
    console.log('Setting up Firebase Remote Config...');
    
    // Get existing template
    const template = await remoteConfig.getTemplate();
    console.log('Found existing template');
    
    // Add/overwrite parameters
    for (const [key, value] of Object.entries(parameters)) {
      template.parameters[key] = value;
    }
    
    // Publish directly
    await remoteConfig.publishTemplate(template);
    console.log('Remote Config template published successfully!');
    
    console.log('\nParameters set:');
    Object.keys(parameters).forEach(key => {
      console.log('  -', key);
    });
    
    process.exit(0);
  } catch (error) {
    console.error('Error setting up Remote Config:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

setupRemoteConfig();