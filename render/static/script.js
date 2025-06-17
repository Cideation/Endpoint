let parsedJSON = null;

function runParser() {
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('jsonPreview');

  if (!fileInput.files.length) {
    preview.textContent = "⚠️ No file selected.";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const raw = JSON.parse(reader.result);
      parsedJSON = cleanJSON(raw);
      preview.textContent = JSON.stringify(parsedJSON, null, 2);
    } catch (e) {
      preview.textContent = "❌ Invalid JSON.";
    }
  };
  reader.readAsText(fileInput.files[0]);
}

function cleanJSON(json) {
  return Array.isArray(json)
    ? json.map(item => sanitize(item))
    : sanitize(json);
}

function sanitize(obj) {
  const clean = {};
  for (const key in obj) {
    const trimmedKey = key.trim().toLowerCase().replace(/\s+/g, "_");
    clean[trimmedKey] = obj[key];
  }
  return clean;
}

function pushToNeo() {
  if (!parsedJSON) {
    alert("No cleaned data to push.");
    return;
  }

  fetch("/api/push-neo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsedJSON)
  })
  .then(res => res.ok ? alert("✅ Pushed to Neo4j!") : alert("❌ Push failed"))
  .catch(() => alert("❌ Network error."));
}