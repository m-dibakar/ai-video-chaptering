import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await axios.post(`${BASE_URL}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const pollJobStatus = async (jobId) => {
  const res = await axios.get(`${BASE_URL}/jobs/${jobId}`);
  return res.data;
};

export const searchVideo = async (query, topK = 3) => {
  const res = await axios.get(`${BASE_URL}/search`, {
    params: { q: query, top_k: topK }
  });
  return res.data;
};
