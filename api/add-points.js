// Import necessary modules
const { initializeApp } = require("firebase/app");
const { getDatabase, ref, get, update } = require("firebase/database");

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBXmdv_Zzf0Q2PUo21zZbwMU9zqaVBJ4GA",
  authDomain: "smartbin-f251c.firebaseapp.com",
  databaseURL: "https://smartbin-f251c-default-rtdb.firebaseio.com",
  projectId: "smartbin-f251c",
  storageBucket: "smartbin-f251c.appspot.com",
  messagingSenderId: "1092374967243",
  appId: "1:1092374967243:web:9da6f10eff2564e8ab3c4e",
  measurementId: "G-C3C5L3XQTN"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// Reference to the target path in Realtime Database
const targetPath = ref(database, "MyUsers/tBoLiXG9dyYmk0AZTL1afRDcLQ92/points/received");

// Function to add 20 points
async function incrementPoints() {
  try {
    const snapshot = await get(targetPath);
    if (snapshot.exists()) {
      const currentPoints = snapshot.val();
      const newPoints = currentPoints + 20;

      // Update points in the database
      await update(ref(database, "MyUsers/tBoLiXG9dyYmk0AZTL1afRDcLQ92/points"), { received: newPoints });
      return { message: `Points updated successfully: ${newPoints}` };
    } else {
      return { error: "No data found at the specified path." };
    }
  } catch (error) {
    return { error: "Error updating points." };
  }
}

// Vercel handler for serverless function
module.exports = async (req, res) => {
  if (req.method === 'POST') {
    const result = await incrementPoints();
    if (result.error) {
      return res.status(500).json({ error: result.error });
    }
    return res.status(200).json({ message: result.message });
  }
  return res.status(405).json({ error: "Method not allowed" });
};
