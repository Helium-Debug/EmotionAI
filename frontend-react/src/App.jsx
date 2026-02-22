import TextEmotion from "./models/TextEmotion";
import AudioEmotion from "./models/AudioEmotion";
import ImageEmotion from "./models/ImageEmotions";
import Fusion from "./models/Fusion";
import "./index.css";

function App() {
  return (
    <div className="app-container">
      <h1>🧠 Multimodal Emotion Detection</h1>

      <div className="card">
        <TextEmotion />
      </div>

      <div className="card">
        <AudioEmotion />
      </div>

      <div className="card">
        <ImageEmotion />
      </div>

      <div className="card">
        <Fusion />
      </div>
    </div>
  );
}

export default App;
