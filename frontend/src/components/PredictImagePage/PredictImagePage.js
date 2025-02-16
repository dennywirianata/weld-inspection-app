// src/components/PredictImagePage/PredictImagePage.js
import React, { useState } from 'react';
import axios from 'axios';
import { Form, Button, Alert, Image } from 'react-bootstrap';
import './PredictImagePage.css';

const PredictImagePage = ({ onResult }) => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [confidence, setConfidence] = useState('');
  const [error, setError] = useState('');
  const [imageUrl, setImageUrl] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
    // Create a URL for the image preview
    if (e.target.files[0]) {
      setImageUrl(URL.createObjectURL(e.target.files[0]));
    } else {
      setImageUrl('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/upload/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data.status);
      setConfidence(response.data.details);
      onResult(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      setResult('Error processing file');
      setConfidence('');
      setError('Error processing file');
    }
  };

  return (
    <div className="predict-image-page">
      <h2>Predict Image</h2>
      <Form onSubmit={handleSubmit}>
        <Form.Group controlId="formFile">
          <Form.Label>Select an image:</Form.Label>
          <Form.Control type="file" accept="image/*" onChange={handleFileChange} />
        </Form.Group>
        <Button variant="primary" type="submit" className="mt-3">
          Inspect
        </Button>
      </Form>

      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
      {result && confidence && imageUrl && (
        <div className="mt-4">
          <h3>Results:</h3>
          <div className={`indicator ${result.toLowerCase()}`}>
            {result === 'Accepted' ? (
              <Alert variant="success" className="indicator-alert">
                <i className="fas fa-check-circle"></i> Accepted
              </Alert>
            ) : (
              <Alert variant="danger" className="indicator-alert">
                <i className="fas fa-times-circle"></i> Rejected
              </Alert>
            )}
          </div>
          <p>Confidence: {confidence}</p>
          <div className="image-preview mt-3">
            <h4>Uploaded Image:</h4>
            <Image src={imageUrl} alt="Uploaded" fluid className="small-image" />
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictImagePage;