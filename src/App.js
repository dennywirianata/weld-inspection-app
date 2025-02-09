// src/App.js
import React, { useState } from 'react';
import UploadForm from './components/UploadForm';
import TrainingForm from './components/TrainingForm';
import './styles/global.css'; // Optional: Import global styles

function App() {
  const [inspectionResult, setInspectionResult] = useState('');

  return (
    <div className="App">
      <h1>Weld Joint Inspection System</h1>
      <UploadForm onResult={setInspectionResult} />
      <div>
        <h3>Inspection Results</h3>
        <p>{inspectionResult}</p>
      </div>
      <TrainingForm onTrainingStatus={(status) => console.log(status)} />
    </div>
  );
}

export default App;