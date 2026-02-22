import { useState } from "react";

function ImageEmotion() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/emotion/image", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <h2>🖼 Image Emotion</h2>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <br />
      <button onClick={handlePredict} disabled={!file}>
        Predict
      </button>

      {result && (
        <div>
          <p>Emotion: {result.emotion}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default ImageEmotion;
