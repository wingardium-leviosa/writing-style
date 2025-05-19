const express = require('express');
const fs = require('fs');
const csv = require('csv-parser');
const cors = require('cors');

const app = express();
app.use(cors());

app.get('/feature-distribution', (req, res) => {
  const results = [];
  fs.createReadStream('./data/2020_feature_extraction_0505.csv')
    .pipe(csv())
    .on('data', (data) => results.push(data))
    .on('end', () => {

      if (results.length === 0) {
        console.log("⚠️ CSV is empty or unreadable.");
        return res.status(500).json({ error: "CSV is empty or unreadable" });
      }

      console.log("✅ Sample row from CSV:", results[0]);

      // 提取需要的列并转 float
      const features = [
        'INTRODUCTION_adv_ratio', 
        'INTRODUCTION_passive_voice_ratio', 
        'INTRODUCTION_present_ratio', 
        'INTRODUCTION_polarity', 
        'INTRODUCTION_dep_pair_prep_VERB'
      ];

      const featureData = {};

      features.forEach((col) => {
        featureData[col] = results
          .map((row) => parseFloat(row[col]))
          .filter((val) => !isNaN(val));
      });

      console.log("Feature data to send:", featureData);

      res.json(featureData);
    
    })
    .on("error", (err) => {
      console.error("❌ Error reading CSV:", err);
      res.status(500).json({ error: "Failed to read CSV" });
    });
});

app.listen(5000, () => {
  console.log('Server listening on http://localhost:5000');
});
