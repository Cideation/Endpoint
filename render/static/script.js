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