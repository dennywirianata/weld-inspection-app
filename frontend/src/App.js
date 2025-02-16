// src/App.js
import React, { useState } from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import Header from './components/Header';
import PredictImagePage from './components/PredictImagePage';
import PredictVideoPage from './components/PredictVideoPage';
import UploadTrainingPage from './components/UploadTrainingPage';
//import UploadForm from './components/UploadForm';
//import TrainingForm from './components/TrainingForm';
import Footer from './components/Footer';
import Background from './components/Background';
import './styles/global.css'; // Optional: Import global styles

function App() {
  const [inspectionResult, setInspectionResult] = useState('');
  const [inspectionConfidence, setInspectionConfidence] = useState('');
  const [trainingStatus, setTrainingStatus] = useState('');

  const handleImageResult = (data) => {
    setInspectionResult(data.status);
    setInspectionConfidence(data.details);
  };

  const handleTrainingStatus = (status) => {
    setTrainingStatus(status);
  };

  
  return (
    <div className="App">
        <Header />
      <Background />
       

      <Tabs defaultActiveKey="predict-image" id="uncontrolled-tab-example" className="mb-3">
        <Tab eventKey="predict-image" title="Predict Image">
          <PredictImagePage onResult={handleImageResult} />
        </Tab>
        <Tab eventKey="predict-video" title="Predict Video">
          <PredictVideoPage onResult={handleImageResult} />
        </Tab>
        <Tab eventKey="upload-training" title="Upload Training Data">
          <UploadTrainingPage onTrainingStatus={handleTrainingStatus} />
        </Tab>
      </Tabs>
      <Footer />
    </div>
        //<h1 className="app-title">Weld Joint Inspection System</h1>
      //<UploadForm onResult={(data) => {
       // setInspectionResult(data.result);
        //setInspectionConfidence(data.confidence);
      //}} />
      
      //<TrainingForm onTrainingStatus={(status) => console.log(status)} />
    //</div>
  );
}

export default App;