const express = require('express');
const { initializeApp } = require("firebase/app");
const { getDatabase, ref, get, update } = require("firebase/database");

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

const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

const targetPath = ref(database, "MyUsers/tBoLiXG9dyYmk0AZTL1afRDcLQ92/points/received");

async function incrementPoints() {
  try {
    const snapshot = await get(targetPath);
    if (snapshot.exists()) {
      const currentPoints = snapshot.val();
      const newPoints = currentPoints + 20;

      await update(ref(database, "MyUsers/tBoLiXG9dyYmk0AZTL1afRDcLQ92/points"), { received: newPoints });
      console.log(`Points updated successfully: ${newPoints}`);
    } else {
      console.log("No data found at the specified path.");
    }
  } catch (error) {
    console.error("Error updating points:", error);
  }
}

const appExpress = express();
const port = 4000;

appExpress.get('/add-points', (req, res) => {
  incrementPoints()
    .then(() => {
      res.status(200).send('Points updated successfully');
    })
    .catch(error => {
      res.status(500).send('Error updating points');
    });
});


appExpress.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
