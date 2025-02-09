// plop-templates\component.hbs
import React, { useState } from 'react';
import axios from 'axios';
import { Form, Button, Alert } from 'react-bootstrap';
import './TrainingForm.css';

const TrainingForm = ({ onResult }) => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
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
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(`Result: ${response.data.status} (Details: ${response.data.details})`);
      onResult(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      setResult('Error processing file');
      setError('Error processing file');
    }
  };

  return (
    <div className="trainingForm-form">
      <h2>Upload Image or Video for Inspection</h2>
      <Form onSubmit={handleSubmit}>
        <Form.Group controlId="formFile">
          <Form.Label>Select an image or video:</Form.Label>
          <Form.Control type="file" onChange={handleFileChange} />
        </Form.Group>
        <Button variant="primary" type="submit" className="mt-3">
          Inspect
        </Button>
      </Form>

      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
      {result && (
        <div className="mt-4">
          <h3>Results:</h3>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

export default TrainingForm;