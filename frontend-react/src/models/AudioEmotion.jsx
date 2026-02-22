import { useState } from "react";

function AudioEmotion() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/emotion/audio", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <h2>🎤 Audio Emotion</h2>

      <input
        type="file"
        accept=".wav"
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

export default AudioEmotion;
