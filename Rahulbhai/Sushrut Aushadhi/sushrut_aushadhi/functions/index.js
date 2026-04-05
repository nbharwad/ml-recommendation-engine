const functions = require('firebase-functions');
const { onDocumentCreated, onDocumentUpdated, onDocumentDeleted } = require('firebase-functions/v2/firestore');
const { onCall } = require('firebase-functions/v2/https');
const { user } = require('firebase-functions/v1/auth');
const admin = require('firebase-admin');

admin.initializeApp();

const db = admin.firestore();
const auth = admin.auth();
const FieldValue = admin.firestore.FieldValue;

const NOTIFICATION_TYPES = {
  order_created: 'order_created',
  order_confirmed: 'order_confirmed',
  order_preparing: 'order_preparing',
  out_for_delivery: 'out_for_delivery',
  order_delivered: 'order_delivered',
  order_cancelled: 'order_cancelled',
};

const NOTIFICATION_TEMPLATES = {
  [NOTIFICATION_TYPES.order_created]: {
    title: 'New Order Received',
    body: (order) => `Order #${order.orderId.substring(0, 8)} from ${order.userName} - Rs.${order.totalAmount}`,
  },
  [NOTIFICATION_TYPES.order_confirmed]: {
    title: 'Order Confirmed',
    body: (order) => `Your order #${order.orderId.substring(0, 8)} has been confirmed!`,
  },
  [NOTIFICATION_TYPES.order_preparing]: {
    title: 'Order Preparing',
    body: (order) => `Your order #${order.orderId.substring(0, 8)} is being prepared.`,
  },
  [NOTIFICATION_TYPES.out_for_delivery]: {
    title: 'Order Out for Delivery',
    body: (order) => `Your order #${order.orderId.substring(0, 8)} is on the way!`,
  },
  [NOTIFICATION_TYPES.order_delivered]: {
    title: 'Order Delivered',
    body: (order) => `Your order #${order.orderId.substring(0, 8)} has been delivered. Thank you!`,
  },
  [NOTIFICATION_TYPES.order_cancelled]: {
    title: 'Order Cancelled',
    body: (order) => `Your order #${order.orderId.substring(0, 8)} has been cancelled.`,
  },
};

const VALID_STATUS_TRANSITIONS = {
  pending: ['confirmed', 'cancelled'],
  confirmed: ['preparing', 'cancelled'],
  preparing: ['outForDelivery', 'cancelled'],
  outForDelivery: ['delivered'],
  delivered: [],
  cancelled: [],
};

async function getAdminTokens() {
  const snapshot = await db.collection('users').where('role', '==', 'admin').get();
  const tokens = [];
  snapshot.forEach(doc => {
    const data = doc.data();
    // Collect all tokens from fcmTokens map (multi-device support)
    if (data.fcmTokens && typeof data.fcmTokens === 'object') {
      tokens.push(...Object.values(data.fcmTokens));
    } else if (data.fcmToken) {
      // Fallback to legacy single-token field
      tokens.push(data.fcmToken);
    }
  });
  return [...new Set(tokens)]; // deduplicate
}

// Returns all FCM tokens for a user across all their devices.
async function getUserTokens(userId) {
  const doc = await db.collection('users').doc(userId).get();
  if (!doc.exists) return [];
  const data = doc.data();
  if (data.fcmTokens && typeof data.fcmTokens === 'object') {
    return [...new Set(Object.values(data.fcmTokens))];
  }
  if (data.fcmToken) return [data.fcmToken];
  return [];
}

async function sendNotification(tokens, title, body, data) {
  if (!tokens || tokens.length === 0) {
    console.log('No tokens to send notification to');
    return { status: 'no_tokens' };
  }

  const messages = tokens.map(token => ({
    token,
    notification: { title, body },
    data: { ...data, timestamp: new Date().toISOString() },
    android: {
      priority: 'high',
      notification: {
        channelId: 'order_updates',
        priority: 'high',
      },
    },
    apns: {
      payload: {
        aps: {
          sound: 'default',
          badge: 1,
        },
      },
    },
  }));

  try {
    const responses = await Promise.all(
      messages.map(msg => admin.messaging().send(msg))
    );
    console.log(`Successfully sent ${responses.length} notifications`);
    return { status: 'sent', count: responses.length };
  } catch (error) {
    console.error('Error sending notifications:', error);
    if (error.code === 'messaging/invalid-registration-token' ||
        error.code === 'messaging/registration-token-not-registered') {
      return { status: 'invalid_token', error: error.message };
    }
    return { status: 'error', error: error.message };
  }
}

async function deleteInvalidToken(userId, token) {
  try {
    const doc = await db.collection('users').doc(userId).get();
    if (!doc.exists) return;
    const data = doc.data();
    const update = {};

    // Remove from legacy single-token field if it matches
    if (data.fcmToken === token) {
      update.fcmToken = FieldValue.delete();
    }

    // Remove matching entry from fcmTokens map
    if (data.fcmTokens && typeof data.fcmTokens === 'object') {
      for (const [deviceId, storedToken] of Object.entries(data.fcmTokens)) {
        if (storedToken === token) {
          update[`fcmTokens.${deviceId}`] = FieldValue.delete();
        }
      }
    }

    if (Object.keys(update).length > 0) {
      await db.collection('users').doc(userId).update(update);
      console.log(`Deleted invalid FCM token for user ${userId}`);
    }
  } catch (error) {
    console.error(`Error deleting invalid token for user ${userId}:`, error);
  }
}

async function sendToTopic(topic, title, body, data) {
  try {
    const response = await admin.messaging().send({
      topic,
      notification: { title, body },
      data: { ...data, timestamp: new Date().toISOString() },
      android: {
        priority: 'high',
        notification: {
          channelId: 'order_updates',
          priority: 'high',
        },
      },
      apns: {
        payload: {
          aps: {
            sound: 'default',
            badge: 1,
          },
        },
      },
    });
    console.log(`Successfully sent to topic ${topic}:`, response);
    return { status: 'sent', messageId: response };
  } catch (error) {
    console.error(`Error sending to topic ${topic}:`, error);
    return { status: 'error', error: error.message };
  }
}

async function decrementStock(items) {
  const batch = db.batch();
  
  for (const item of items) {
    const medicineRef = db.collection('medicines').doc(item.medicineId);
    batch.update(medicineRef, {
      stock: FieldValue.increment(-item.quantity),
    });
  }
  
  await batch.commit();
  console.log(`Stock decremented for ${items.length} items`);
}

async function restoreStock(items) {
  const batch = db.batch();
  
  for (const item of items) {
    const medicineRef = db.collection('medicines').doc(item.medicineId);
    batch.update(medicineRef, {
      stock: FieldValue.increment(item.quantity),
    });
  }
  
  await batch.commit();
  console.log(`Stock restored for ${items.length} items`);
}

function validateStatusTransition(currentStatus, newStatus) {
  const allowedTransitions = VALID_STATUS_TRANSITIONS[currentStatus] || [];
  return allowedTransitions.includes(newStatus);
}

exports.validateAndCalculateOrder = onDocumentCreated(
  'orders/{orderId}',
  async (event) => {
    const orderData = event.data.data();
    const orderId = event.params.orderId;
    
    console.log(`Validating order: ${orderId}`);
    
    let calculatedTotal = 0;
    const items = orderData.items || [];
    
    if (items.length === 0) {
      return event.data.ref.delete().then(() => {
        console.log(`Order ${orderId} deleted: no items`);
        return { status: 'deleted', reason: 'no_items' };
      });
    }

    // Fetch actual prices from Firestore — never trust client-submitted prices
    for (const item of items) {
      const medicineId = item.medicineId;
      if (!medicineId) {
        console.error(`Order ${orderId}: item missing medicineId, rejecting order`);
        return event.data.ref.delete().then(() => {
          return { status: 'deleted', reason: 'missing_medicine_id' };
        });
      }

      const medicineDoc = await db.collection('medicines').doc(medicineId).get();
      if (!medicineDoc.exists) {
        console.error(`Order ${orderId}: medicine ${medicineId} not found, rejecting order`);
        return event.data.ref.delete().then(() => {
          return { status: 'deleted', reason: `medicine_not_found:${medicineId}` };
        });
      }

      const medicineData = medicineDoc.data();
      if (medicineData.isActive === false) {
        console.error(`Order ${orderId}: medicine ${medicineId} is inactive, rejecting order`);
        return event.data.ref.delete().then(() => {
          return { status: 'deleted', reason: `medicine_inactive:${medicineId}` };
        });
      }

      const actualPrice = parseFloat(medicineData.price) || 0;
      const quantity = parseInt(item.quantity) || 0;
      calculatedTotal += actualPrice * quantity;
    }

    const submittedTotal = parseFloat(orderData.totalAmount) || 0;
    const difference = Math.abs(calculatedTotal - submittedTotal);

    if (difference > 1) {
      console.log(`Order ${orderId}: total mismatch. Submitted: ${submittedTotal}, Calculated: ${calculatedTotal}`);

      return event.data.ref.update({
        totalAmount: calculatedTotal,
        originalSubmittedAmount: submittedTotal,
        isTotalValidated: true,
        validatedAt: FieldValue.serverTimestamp(),
        validationNote: `Total was corrected from ${submittedTotal} to ${calculatedTotal}`
      }).then(() => {
        console.log(`Order ${orderId} total corrected to ${calculatedTotal}`);
        return { status: 'corrected', original: submittedTotal, corrected: calculatedTotal };
      });
    }

    return event.data.ref.update({
      totalAmount: calculatedTotal,
      isTotalValidated: true,
      validatedAt: FieldValue.serverTimestamp()
    }).then(() => {
      console.log(`Order ${orderId} validated successfully. Total: ${calculatedTotal}`);
      return { status: 'valid', total: calculatedTotal };
    }).catch(error => {
      console.error(`Error validating order ${orderId}:`, error);
      return { status: 'error', error: error.message };
    });
  }
);

exports.validateAndCalculateLabOrder = onDocumentCreated(
  'labOrders/{orderId}',
  async (event) => {
    const orderData = event.data.data();
    const orderId = event.params.orderId;

    console.log(`Validating lab order: ${orderId}`);

    const tests = orderData.tests || [];

    if (tests.length === 0) {
      return event.data.ref.delete().then(() => {
        console.log(`Lab order ${orderId} deleted: no tests`);
        return { status: 'deleted', reason: 'no_tests' };
      });
    }

    let calculatedTotal = 0;

    // Fetch actual prices from Firestore — never trust client-submitted prices
    for (const test of tests) {
      const testId = test.testId;
      if (!testId) {
        console.error(`Lab order ${orderId}: test item missing testId, rejecting order`);
        return event.data.ref.delete().then(() => {
          return { status: 'deleted', reason: 'missing_test_id' };
        });
      }

      const testDoc = await db.collection('lab_tests').doc(testId).get();
      if (!testDoc.exists) {
        console.error(`Lab order ${orderId}: test ${testId} not found, rejecting order`);
        return event.data.ref.delete().then(() => {
          return { status: 'deleted', reason: `test_not_found:${testId}` };
        });
      }

      const testData = testDoc.data();
      if (testData.active === false) {
        console.error(`Lab order ${orderId}: test ${testId} is inactive, rejecting order`);
        return event.data.ref.delete().then(() => {
          return { status: 'deleted', reason: `test_inactive:${testId}` };
        });
      }

      const actualPrice = parseFloat(testData.price) || 0;
      calculatedTotal += actualPrice;
    }

    const submittedTotal = parseFloat(orderData.totalAmount) || 0;
    const difference = Math.abs(calculatedTotal - submittedTotal);

    if (difference > 1) {
      console.log(`Lab order ${orderId}: total mismatch. Submitted: ${submittedTotal}, Calculated: ${calculatedTotal}`);

      return event.data.ref.update({
        totalAmount: calculatedTotal,
        originalSubmittedAmount: submittedTotal,
        isTotalValidated: true,
        validatedAt: FieldValue.serverTimestamp(),
        validationNote: `Total was corrected from ${submittedTotal} to ${calculatedTotal}`
      }).then(() => {
        console.log(`Lab order ${orderId} total corrected to ${calculatedTotal}`);
        return { status: 'corrected', original: submittedTotal, corrected: calculatedTotal };
      });
    }

    return event.data.ref.update({
      totalAmount: calculatedTotal,
      isTotalValidated: true,
      validatedAt: FieldValue.serverTimestamp()
    }).then(() => {
      console.log(`Lab order ${orderId} validated successfully. Total: ${calculatedTotal}`);
      return { status: 'valid', total: calculatedTotal };
    }).catch(error => {
      console.error(`Error validating lab order ${orderId}:`, error);
      return { status: 'error', error: error.message };
    });
  }
);

exports.validatePrescription = onDocumentCreated(
  'prescriptions/{prescriptionId}',
  async (event) => {
    const prescriptionData = event.data.data();
    const prescriptionId = event.params.prescriptionId;
    
    const authUid = prescriptionData.userId;
    
    console.log(`Validating prescription: ${prescriptionId}`);
    console.log(`Authenticated UID: ${authUid}`);
    
    if (!authUid) {
      console.error(`SECURITY: Prescription ${prescriptionId} created without userId!`);
      return event.data.ref.delete().then(() => {
        console.log(`Invalid prescription ${prescriptionId} deleted`);
        return { status: 'rejected', reason: 'no_user_id' };
      }).catch(error => {
        console.error(`Error deleting invalid prescription:`, error);
        return { status: 'error', error: error.message };
      });
    }
    
    const requiredFields = ['userId', 'userName', 'imageUrl'];
    const missingFields = requiredFields.filter(field => !prescriptionData[field]);
    
    if (missingFields.length > 0) {
      console.log(`Prescription ${prescriptionId} missing fields: ${missingFields.join(', ')}`);
    }
    
    console.log(`Prescription ${prescriptionId} validated successfully for user: ${authUid}`);
    return { status: 'valid', userId: authUid };
  }
);

exports.onPrescriptionReview = onDocumentUpdated(
  'prescriptions/{prescriptionId}',
  async (event) => {
    const newData = event.data.after.data();
    const previousData = event.data.before.data();
    const prescriptionId = event.params.prescriptionId;
    
    if (previousData.status !== newData.status) {
      console.log(`Prescription ${prescriptionId} status changed: ${previousData.status} -> ${newData.status}`);
      
      await db.collection('admin_audit_log').add({
        action: 'prescription_review',
        performedBy: newData.reviewedBy || 'system',
        targetId: prescriptionId,
        targetCollection: 'prescriptions',
        previousStatus: previousData.status,
        newStatus: newData.status,
        timestamp: FieldValue.serverTimestamp(),
      });
      
      return { status: 'logged', prescriptionId };
    }
    
    return Promise.resolve({ status: 'no_change' });
  }
);

exports.setUserRole = onCall(async (request) => {
  const { uid, role } = request.data;

  if (!request.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be authenticated');
  }

  const adminRole = request.auth.token?.role;
  const isAdmin = adminRole === 'admin';

  if (!isAdmin) {
    throw new functions.https.HttpsError('permission-denied', 'Only admins can set roles');
  }

  if (!uid || !role) {
    throw new functions.https.HttpsError('invalid-argument', 'uid and role are required');
  }

  const validRoles = ['admin', 'delivery', 'customer'];
  if (!validRoles.includes(role)) {
    throw new functions.https.HttpsError('invalid-argument', 'Invalid role');
  }

  await auth.setCustomUserClaims(uid, { role: role });
  await db.collection('users').doc(uid).update({
    role: role,
    isAdmin: role === 'admin',
    updatedAt: FieldValue.serverTimestamp()
  });

  return { success: true, uid, role };
});

exports.revokeAdminRole = onCall(async (request) => {
  const { uid } = request.data;

  if (!request.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be authenticated');
  }

  const adminRole = request.auth.token?.role;
  const isAdmin = adminRole === 'admin';

  if (!isAdmin) {
    throw new functions.https.HttpsError('permission-denied', 'Only admins can revoke roles');
  }

  if (!uid) {
    throw new functions.https.HttpsError('invalid-argument', 'uid is required');
  }

  await auth.setCustomUserClaims(uid, { role: 'customer' });
  await db.collection('users').doc(uid).update({
    role: 'customer',
    isAdmin: false,
    updatedAt: FieldValue.serverTimestamp()
  });

  return { success: true, uid, role: 'customer' };
});

exports.updateOrderStatus = onCall(async (request) => {
  const { orderId, newStatus, statusNote } = request.data;

  if (!request.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be authenticated');
  }

  const userRole = request.auth.token?.role;
  const isAdmin = userRole === 'admin';
  const isDelivery = userRole === 'delivery';

  if (!isAdmin && !isDelivery) {
    throw new functions.https.HttpsError('permission-denied', 'Only admins or delivery staff can update order status');
  }

  if (!orderId || !newStatus) {
    throw new functions.https.HttpsError('invalid-argument', 'orderId and newStatus are required');
  }
  
  const validAdminTransitions = {
    pending: ['confirmed', 'cancelled'],
    confirmed: ['preparing', 'cancelled'],
    preparing: ['outForDelivery', 'cancelled'],
    outForDelivery: ['delivered'],
    delivered: [],
    cancelled: [],
  };
  
  const validDeliveryTransitions = {
    outForDelivery: ['delivered'],
  };
  
  const orderDoc = await db.collection('orders').doc(orderId).get();
  
  if (!orderDoc.exists) {
    throw new functions.https.HttpsError('not-found', 'Order not found');
  }
  
  const currentStatus = orderDoc.data().status;
  let allowedTransitions;
  
  if (isDelivery) {
    allowedTransitions = validDeliveryTransitions[currentStatus] || [];
    if (!allowedTransitions.includes(newStatus)) {
      throw new functions.https.HttpsError('permission-denied', 'Delivery staff can only mark orders as delivered');
    }
  } else {
    allowedTransitions = validAdminTransitions[currentStatus] || [];
    if (!allowedTransitions.includes(newStatus)) {
      throw new functions.https.HttpsError('invalid-argument', `Invalid status transition from ${currentStatus} to ${newStatus}`);
    }
  }
  
  const statusHistoryEntry = {
    status: newStatus,
    timestamp: FieldValue.serverTimestamp(),
    updatedBy: request.auth.uid,
    role: userRole,
    note: statusNote || '',
  };
  
  await db.collection('orders').doc(orderId).update({
    status: newStatus,
    updatedAt: FieldValue.serverTimestamp(),
    updatedBy: request.auth.uid,
    statusNote: statusNote || '',
    statusHistory: FieldValue.arrayUnion(statusHistoryEntry),
  });
  
  await db.collection('admin_audit_log').add({
    orderId: orderId,
    previousStatus: currentStatus,
    newStatus: newStatus,
    timestamp: FieldValue.serverTimestamp(),
    changeType: 'status_change',
    updatedBy: request.auth.uid,
    userRole: userRole,
    validTransition: true,
  });
  
  return { success: true, orderId, previousStatus: currentStatus, newStatus };
});

exports.syncAdminClaims = onCall(async (request) => {
  if (!request.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be authenticated');
  }

  const adminRole = request.auth.token?.role;
  const isAdmin = adminRole === 'admin';
  
  if (!isAdmin) {
    throw new functions.https.HttpsError('permission-denied', 'Only admins can sync claims');
  }
  
  const snapshot = await db.collection('users').where('role', '==', 'admin').get();
  let synced = 0;
  let errors = [];
  
  for (const doc of snapshot.docs) {
    try {
      await admin.auth().setCustomUserClaims(doc.id, { role: 'admin' });
      synced++;
    } catch (error) {
      errors.push({ uid: doc.id, error: error.message });
    }
  }
  
  return { success: true, synced, errors: errors.length > 0 ? errors : null };
});

exports.migrateUserRoles = onCall(async (request) => {
  const { secret } = request.data;
  
  if (secret !== 'SUSHRUT_MIGRATE_2026') {
    throw new functions.https.HttpsError('invalid-argument', 'Invalid secret');
  }
  
  const snapshot = await db.collection('users').get();
  let migrated = 0;
  let errors = [];
  
  for (const doc of snapshot.docs) {
    try {
      const userData = doc.data();
      const isAdmin = userData.isAdmin === true;
      const role = isAdmin ? 'admin' : 'customer';
      
      await auth.setCustomUserClaims(doc.id, { role: role });
      
      await doc.ref.update({ role: role });
      migrated++;
    } catch (error) {
      errors.push({ uid: doc.id, error: error.message });
    }
  }
  
  return { success: true, migrated, errors: errors.length > 0 ? errors : null };
});

exports.bootstrapFirstAdmin = onCall(async (request) => {
  const { uid, secret } = request.data;
  
  if (secret !== 'SUSHRUT_BOOTSTRAP_2026') {
    throw new functions.https.HttpsError('invalid-argument', 'Invalid secret');
  }
  
  if (!uid) {
    throw new functions.https.HttpsError('invalid-argument', 'uid is required');
  }
  
  await auth.setCustomUserClaims(uid, { role: 'admin' });
  await db.collection('users').doc(uid).update({ 
    role: 'admin',
    isAdmin: true,
    updatedAt: FieldValue.serverTimestamp()
  });
  
  return { success: true, uid, role: 'admin' };
});

async function checkServerRateLimit(userId, actionType, maxPerDay) {
  const today = new Date();
  const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const endOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 1);
  
  const snapshot = await db.collection(actionType)
    .where('userId', '==', userId)
    .where('createdAt', '>=', startOfDay)
    .where('createdAt', '<', endOfDay)
    .count()
    .get();
  
  return (snapshot.data().count || 0) < maxPerDay;
}

exports.checkOrderRateLimit = onCall(async (request) => {
  if (!request.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be authenticated');
  }
  
  const userId = request.auth.uid;
  const maxOrdersPerDay = 10;
  
  const canOrder = await checkServerRateLimit(userId, 'orders', maxOrdersPerDay);
  
  if (!canOrder) {
    throw new functions.https.HttpsError('resource-exhausted', 'Daily order limit reached. Please try again tomorrow.');
  }
  
  return { success: true, remaining: maxOrdersPerDay };
});

exports.checkPrescriptionRateLimit = onCall(async (request) => {
  if (!request.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be authenticated');
  }
  
  const userId = request.auth.uid;
  const maxPrescriptionsPerDay = 5;
  
  const canUpload = await checkServerRateLimit(userId, 'prescriptions', maxPrescriptionsPerDay);
  
  if (!canUpload) {
    throw new functions.https.HttpsError('resource-exhausted', 'Daily prescription upload limit reached. Please try again tomorrow.');
  }
  
  return { success: true, remaining: maxPrescriptionsPerDay };
});

exports.onUserCreated = user().onCreate(async (user) => {
  const userDoc = await db.collection('users').doc(user.uid).get();
  
  if (!userDoc.exists) {
    await db.collection('users').doc(user.uid).set({
      name: user.displayName || '',
      phone: user.phoneNumber || '',
      createdAt: FieldValue.serverTimestamp(),
      isAdmin: false,
      role: 'customer',
      isActive: true,
    });
    console.log(`Created default user document for ${user.uid}`);
  }

  await auth.setCustomUserClaims(user.uid, { role: 'customer' });
  console.log(`Set default custom claims for user ${user.uid}`);
});

exports.logAdminAction = onDocumentUpdated(
  'orders/{orderId}',
  async (event) => {
    const newData = event.data.after.data();
    const previousData = event.data.before.data();
    const orderId = event.params.orderId;
    
    if (previousData.status !== newData.status) {
      console.log(`Order ${orderId} status changed: ${previousData.status} -> ${newData.status}`);
      
      const currentStatus = previousData.status;
      const newStatus = newData.status;
      const isValidTransition = validateStatusTransition(currentStatus, newStatus);
      
      if (!isValidTransition) {
        console.error(`Invalid status transition: ${currentStatus} -> ${newStatus}`);
        await event.data.ref.update({
          status: currentStatus,
          statusTransitionError: `Invalid transition from ${currentStatus} to ${newStatus}`,
        });
        return { status: 'rejected', reason: 'invalid_status_transition' };
      }
      
      const statusHistoryEntry = {
        status: newStatus,
        timestamp: FieldValue.serverTimestamp(),
        updatedBy: newData.updatedBy || 'system',
        note: newData.statusNote || '',
      };
      
      await event.data.ref.update({
        statusHistory: FieldValue.arrayUnion(statusHistoryEntry),
        updatedAt: FieldValue.serverTimestamp(),
      });
      
      await db.collection('admin_audit_log').add({
        orderId: orderId,
        previousStatus: currentStatus,
        newStatus: newStatus,
        timestamp: FieldValue.serverTimestamp(),
        changeType: 'status_change',
        updatedBy: newData.updatedBy || 'system',
        validTransition: isValidTransition,
      });
      
      return { status: 'logged' };
    }
    
    if (previousData.items !== newData.items) {
      console.log(`Order ${orderId} items modified`);
      
      await db.collection('admin_audit_log').add({
        orderId: orderId,
        timestamp: FieldValue.serverTimestamp(),
        changeType: 'items_modified',
        updatedBy: newData.updatedBy || 'system',
      });
    }
    
    return Promise.resolve({ status: 'no_change' });
  }
);

exports.handleOrderCancellation = onDocumentUpdated(
  'orders/{orderId}',
  async (event) => {
    const newData = event.data.after.data();
    const previousData = event.data.before.data();
    const orderId = event.params.orderId;
    
    if (previousData.status !== 'cancelled' && newData.status === 'cancelled') {
      console.log(`Order ${orderId} cancelled - restoring stock`);
      
      const items = previousData.items || [];
      if (items.length > 0) {
        await restoreStock(items);
      }
      
      await db.collection('admin_audit_log').add({
        orderId: orderId,
        timestamp: FieldValue.serverTimestamp(),
        changeType: 'order_cancelled',
        restoredItems: items.length,
        updatedBy: newData.updatedBy || 'system',
      });
      
      return { status: 'stock_restored', orderId };
    }
    
    return Promise.resolve({ status: 'no_change' });
  }
);

exports.decrementStockOnOrder = onDocumentCreated(
  'orders/{orderId}',
  async (event) => {
    const orderData = event.data.data();
    const orderId = event.params.orderId;
    
    const items = orderData.items || [];
    
    if (items.length > 0) {
      await decrementStock(items);
      console.log(`Stock decremented for order ${orderId}`);
    }
    
    return { status: 'processed', orderId };
  }
);

exports.sendNewOrderNotification = onDocumentCreated(
  'orders/{orderId}',
  async (event) => {
    const orderData = event.data.data();
    const orderId = event.params.orderId;
    
    console.log(`Processing new order notification: ${orderId}`);
    
    const template = NOTIFICATION_TEMPLATES[NOTIFICATION_TYPES.order_created];
    const title = template.title;
    const body = template.body(orderData);
    const data = {
      type: NOTIFICATION_TYPES.order_created,
      orderId: orderId,
    };
    
    const adminTokens = await getAdminTokens();
    
    if (adminTokens.length > 0) {
      await sendNotification(adminTokens, title, body, data);
    } else {
      await sendToTopic('admin_orders', title, body, data);
    }
    
    const adminSnapshot = await db.collection('users').where('role', '==', 'admin').get();
    const notificationPromises = adminSnapshot.docs.map(async (adminDoc) => {
      await db.collection('notifications').add({
        userId: adminDoc.id,
        title: title,
        body: body,
        type: NOTIFICATION_TYPES.order_created,
        orderId: orderId,
        isRead: false,
        createdAt: FieldValue.serverTimestamp(),
      });
    });
    await Promise.all(notificationPromises);
    
    return { status: 'processed', orderId };
  }
);

exports.sendLabOrderNotification = onDocumentCreated(
  'notifications/{notificationId}',
  async (event) => {
    const notificationData = event.data.data();
    const notificationId = event.params.notificationId;
    
    console.log(`Processing lab order notification: ${notificationId}`);
    
    const userId = notificationData.userId;
    const title = notificationData.title;
    const body = notificationData.body;
    const type = notificationData.type;
    const orderId = notificationData.orderId;
    
    if (!userId || !title || !body) {
      console.log('Missing required fields for notification');
      return { status: 'skipped', reason: 'missing_fields' };
    }
    
    const data = {
      type: type || 'lab_notification',
      orderId: orderId || '',
      timestamp: new Date().toISOString(),
    };
    
    const userTokens = await getUserTokens(userId);

    if (userTokens.length > 0) {
      const result = await sendNotification(userTokens, title, body, data);

      if (result.status === 'invalid_token') {
        console.log(`Invalid token for user ${userId} - cleaning up`);
        for (const token of userTokens) {
          await deleteInvalidToken(userId, token);
        }
      }
    } else {
      await sendToTopic(`customer_${userId}`, title, body, data);
    }

    return { status: 'processed', notificationId };
  }
);

exports.sendOrderStatusNotification = onDocumentUpdated(
  'orders/{orderId}',
  async (event) => {
    const newData = event.data.after.data();
    const previousData = event.data.before.data();
    const orderId = event.params.orderId;
    
    if (previousData.status === newData.status) {
      return { status: 'no_change' };
    }
    
    console.log(`Processing order status notification: ${orderId}, ${previousData.status} -> ${newData.status}`);
    
    let notificationType;
    switch (newData.status) {
      case 'confirmed':
        notificationType = NOTIFICATION_TYPES.order_confirmed;
        break;
      case 'preparing':
        notificationType = NOTIFICATION_TYPES.order_preparing;
        break;
      case 'outForDelivery':
        notificationType = NOTIFICATION_TYPES.out_for_delivery;
        break;
      case 'delivered':
        notificationType = NOTIFICATION_TYPES.order_delivered;
        break;
      case 'cancelled':
        notificationType = NOTIFICATION_TYPES.order_cancelled;
        break;
      default:
        console.log(`No notification type for status: ${newData.status}`);
        return { status: 'skipped', reason: 'no_notification_type' };
    }
    
    const template = NOTIFICATION_TEMPLATES[notificationType];
    const title = template.title;
    const body = template.body(newData);
    const data = {
      type: notificationType,
      orderId: orderId,
    };
    
    const userId = newData.userId;
    const userTokens = await getUserTokens(userId);

    if (userTokens.length > 0) {
      const result = await sendNotification(userTokens, title, body, data);

      if (result.status === 'invalid_token') {
        console.log(`Invalid token for user ${userId} - cleaning up`);
        for (const token of userTokens) {
          await deleteInvalidToken(userId, token);
        }
      }
    } else {
      await sendToTopic(`customer_${userId}`, title, body, data);
    }
    
    await db.collection('notifications').add({
      userId: userId,
      title: title,
      body: body,
      type: notificationType,
      orderId: orderId,
      isRead: false,
      createdAt: FieldValue.serverTimestamp(),
    });
    
    if (newData.status === 'delivered') {
      await event.data.after.ref.update({
        deliveredAt: FieldValue.serverTimestamp(),
      });
    }
    
    return { status: 'processed', orderId, type: notificationType };
  }
);
