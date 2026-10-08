
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api';

async function request(path, options = {}) {
  const token = localStorage.getItem('cp_token');
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

export const api = {
  register: (body) => request('/auth/register', { method: 'POST', body: JSON.stringify(body) }),
  login: (body) => request('/auth/login', { method: 'POST', body: JSON.stringify(body) }),
  me: () => request('/auth/me'),
  models: () => request('/models'),
  predict: (body) => request('/predict', { method: 'POST', body: JSON.stringify(body) }),
  compare: (body) => request('/predict/compare', { method: 'POST', body: JSON.stringify(body) }),
  history: () => request('/history'),
  historyItem: (id) => request(`/history/${id}`),
  profile: (full_name) => request(`/profile?full_name=${encodeURIComponent(full_name)}`, { method: 'PUT' }),
};
