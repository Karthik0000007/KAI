import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

class ApiService {
  constructor() {
    this.token = null;
    this.ws = null;
    this.wsCallbacks = [];
  }

  setToken(token) {
    this.token = token;
  }

  getHeaders() {
    return {
      'Content-Type': 'application/json',
      ...(this.token && { 'Authorization': `Bearer ${this.token}` })
    };
  }

  // Authentication
  async login(username, password) {
    const response = await axios.post(`${API_BASE_URL}/auth/login`, {
      username,
      password
    });
    return response.data;
  }

  async logout() {
    if (!this.token) return;
    await axios.post(`${API_BASE_URL}/auth/logout`, {}, {
      headers: this.getHeaders()
    });
  }

  // Health Data
  async getMoodTrend(days = 30) {
    const response = await axios.get(`${API_BASE_URL}/health/mood`, {
      params: { days },
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getSleepTrend(days = 30) {
    const response = await axios.get(`${API_BASE_URL}/health/sleep`, {
      params: { days },
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getEnergyTrend(days = 30) {
    const response = await axios.get(`${API_BASE_URL}/health/energy`, {
      params: { days },
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getEmotionDistribution(days = 30) {
    const response = await axios.get(`${API_BASE_URL}/health/emotion-distribution`, {
      params: { days },
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getVitalSigns(days = 7) {
    const response = await axios.get(`${API_BASE_URL}/health/vital-signs`, {
      params: { days },
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getProactiveAlerts() {
    const response = await axios.get(`${API_BASE_URL}/health/proactive-alerts`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async acknowledgeAlert(alertId) {
    const response = await axios.post(
      `${API_BASE_URL}/health/alerts/${alertId}/acknowledge`,
      {},
      { headers: this.getHeaders() }
    );
    return response.data;
  }

  async getHealthStatistics() {
    const response = await axios.get(`${API_BASE_URL}/health/statistics`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async exportData(format = 'csv') {
    const response = await axios.get(`${API_BASE_URL}/data/export`, {
      params: { format },
      headers: this.getHeaders(),
      responseType: 'blob'
    });
    return response.data;
  }

  // WebSocket for live updates
  connectWebSocket(onMessage) {
    if (!this.token) return;

    this.ws = new WebSocket(`ws://127.0.0.1:8000/ws?token=${this.token}`);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
      this.wsCallbacks.forEach(callback => callback(data));
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (this.token) {
          this.connectWebSocket(onMessage);
        }
      }, 5000);
    };
  }

  disconnectWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  onWebSocketMessage(callback) {
    this.wsCallbacks.push(callback);
  }
}

export const api = new ApiService();
