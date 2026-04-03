const admin = require("firebase-admin");
const csv = require("csvtojson");
const serviceAccount = require("./serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

// ← Change this to where it stopped (e.g. 19500)
const SKIP_FIRST = 19500;

// Delay between batches to avoid quota
const DELAY_MS = 2000;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function uploadMedicines() {
  const allMedicines = await csv().fromFile("./medicines.csv");
  
  // Skip already uploaded
  const medicines = allMedicines.slice(SKIP_FIRST);
  console.log(`Resuming from ${SKIP_FIRST}...`);
  console.log(`Remaining to upload: ${medicines.length}`);

  const batchSize = 200; // Smaller batch to avoid quota
  let count = SKIP_FIRST;

  for (let i = 0; i < medicines.length; i += batchSize) {
    const batch = db.batch();
    const chunk = medicines.slice(i, i + batchSize);

    chunk.forEach((med) => {
      const ref = db.collection("medicines").doc();
      batch.set(ref, {
        name: med["name"] || "",
        manufacturer: med["manufacturer_name"] || "",
        price: parseFloat(med["price(₹)"]) || 0,
        mrp: parseFloat(med["price(₹)"]) || 0,
        unit: med["pack_size_label"] || "strip",
        composition: med["short_composition1"] || "",
        category: detectCategory(med["name"]),
        requiresPrescription: detectRx(med["name"]),
        isActive: med["Is_discontinued"] === "FALSE",
        stock: 50,
        imageUrl: "",
        createdAt: admin.firestore.FieldValue.serverTimestamp()
      });
    });

    await batch.commit();
    count += chunk.length;
    console.log(`Uploaded ${count} medicines...`);

    // Wait 2 seconds between batches
    await sleep(DELAY_MS);
  }

  console.log("✅ All medicines uploaded successfully!");
}

function detectCategory(name) {
  const n = name.toLowerCase();
  if (n.includes("paracetamol") || n.includes("dolo") || n.includes("crocin")) return "fever";
  if (n.includes("amoxicillin") || n.includes("azithromycin") || n.includes("ciprofloxacin")) return "antibiotics";
  if (n.includes("vitamin") || n.includes("calcium") || n.includes("zinc")) return "vitamins";
  if (n.includes("ibuprofen") || n.includes("diclofenac") || n.includes("pain")) return "pain_relief";
  if (n.includes("metformin") || n.includes("insulin") || n.includes("gluco")) return "diabetes";
  if (n.includes("atorvastatin") || n.includes("amlodipine")) return "heart";
  if (n.includes("cream") || n.includes("lotion") || n.includes("gel")) return "skin";
  return "other";
}

function detectRx(name) {
  const rxMeds = ["amoxicillin","azithromycin","metformin","atorvastatin",
                  "amlodipine","ciprofloxacin","metoprolol","omeprazole"];
  const n = name.toLowerCase();
  return rxMeds.some(rx => n.includes(rx));
}

uploadMedicines().catch(console.error);