# SOS Admin UI

This is a lightweight HTML/JS/CSS interface for managing your SOS system, designed to be deployed on Railway.

## 📁 Features

- **📤 Upload JSON** — Select a raw JSON file for processing
- **🧹 Parse & Clean** — Normalizes keys and formats data
- **👁 Preview Dictionary** — Displays cleaned JSON in real-time
- **🚀 Push to Neo4j** — Sends the data to your Neo4j API endpoint

## 🧪 Local Testing

Open `index.html` in your browser and test interactions.

## 🚀 Deployment (Railway)

1. Push this folder to a GitHub repo
2. Link the repo on Railway (as a **Static Project**)
3. Make sure to set your `/api/push-neo` route in your backend, or mock it for now

## 📌 Notes

- All keys in JSON will be trimmed, lowercased, and space-replaced (`"Node ID"` → `node_id`)
- Make sure your backend has CORS enabled if hosted on another domain

---

© 2025 SOS Project Admin Panel
