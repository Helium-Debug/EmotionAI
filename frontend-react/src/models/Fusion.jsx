import { useState } from "react";

function Fusion() {
  const [text, setText] = useState("");
  const [audio, setAudio] = useState(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleFusion = async () => {
    const formData = new FormData();

    if (text) formData.append("text", text);
    if (audio) formData.append("audio", audio);
    if (image) formData.append("image", image);

    const response = await fetch("http://127.0.0.1:8000/emotion/fusion", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <h2>🔷 Multimodal Fusion</h2>

      <textarea
        rows="3"
        cols="50"
        placeholder="Optional text..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <br />
      <input type="file" accept=".wav" onChange={(e) => setAudio(e.target.files[0])} />
      <br />
      <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} />
      <br />

      <button onClick={handleFusion}>Run Fusion</button>

      {result && (
        <div>
          <p>Final Emotion: {result.final_emotion}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default Fusion;
