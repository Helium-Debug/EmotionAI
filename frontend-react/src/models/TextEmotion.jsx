import { useState } from "react";

function TextEmotion() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const response = await fetch("http://127.0.0.1:8000/emotion/text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <h2>📝 Text Emotion</h2>

      <textarea
        rows="4"
        cols="50"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text..."
      />

      <br />
      <button onClick={handlePredict}>Predict</button>

      {result && (
        <div>
          <p>Emotion: {result.emotion}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default TextEmotion;
