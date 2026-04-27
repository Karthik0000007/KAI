import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import './Dashboard.css';
import TrendChart from './charts/TrendChart';
import EmotionPieChart from './charts/EmotionPieChart';
import VitalSignsChart from './charts/VitalSignsChart';
import CorrelationChart from './charts/CorrelationChart';
import CalendarView from './CalendarView';
import AlertsPanel from './AlertsPanel';
import StatisticsPanel from './StatisticsPanel';

function Dashboard({ token, onLogout }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [dateRange, setDateRange] = useState(30);
  const [moodData, setMoodData] = useState(null);
  const [sleepData, setSleepData] = useState(null);
  const [energyData, setEnergyData] = useState(null);
  const [emotionData, setEmotionData] = useState(null);
  const [vitalData, setVitalData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState('normal');
  const [textSize, setTextSize] = useState('normal');

  useEffect(() => {
    api.setToken(token);
    loadData();
    
    // Connect WebSocket for live updates
    api.connectWebSocket((data) => {
      console.log('Live update received:', data);
      loadData(); // Reload data when updates arrive
    });

    return () => {
      api.disconnectWebSocket();
    };
  }, [token, dateRange]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [mood, sleep, energy, emotion, vital, alertsData, stats] = await Promise.all([
        api.getMoodTrend(dateRange),
        api.getSleepTrend(dateRange),
        api.getEnergyTrend(dateRange),
        api.getEmotionDistribution(dateRange),
        api.getVitalSigns(Math.min(dateRange, 7)),
        api.getProactiveAlerts(),
        api.getHealthStatistics()
      ]);

      setMoodData(mood);
      setSleepData(sleep);
      setEnergyData(energy);
      setEmotionData(emotion);
      setVitalData(vital);
      setAlerts(alertsData);
      setStatistics(stats);
    } catch (error) {
      console.error('Error loading data:', error);
      if (error.response?.status === 401) {
        onLogout();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledgeAlert = async (alertId) => {
    try {
      await api.acknowledgeAlert(alertId);
      loadData();
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    }
  };

  const handleExportCSV = async () => {
    try {
      const blob = await api.exportData('csv');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aegis-health-data-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting data:', error);
    }
  };

  const toggleTheme = () => {
    const newTheme = theme === 'normal' ? 'high-contrast' : 'normal';
    setTheme(newTheme);
    document.body.className = newTheme;
  };

  const toggleTextSize = () => {
    const newSize = textSize === 'normal' ? 'large-text' : 'normal';
    setTextSize(newSize);
    if (newSize === 'large-text') {
      document.body.classList.add('large-text');
    } else {
      document.body.classList.remove('large-text');
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <a href="#main-content" className="skip-link">Skip to main content</a>
      
      <header className="dashboard-header">
        <h1>Aegis Health Dashboard</h1>
        <div className="header-controls">
          <button 
            onClick={toggleTheme} 
            className="btn btn-secondary"
            aria-label="Toggle high contrast theme"
          >
            {theme === 'normal' ? '🌙 High Contrast' : '☀️ Normal'}
          </button>
          <button 
            onClick={toggleTextSize} 
            className="btn btn-secondary"
            aria-label="Toggle large text"
          >
            {textSize === 'normal' ? '🔍 Large Text' : '🔍 Normal Text'}
          </button>
          <button onClick={handleExportCSV} className="btn btn-secondary">
            📥 Export CSV
          </button>
          <button onClick={onLogout} className="btn btn-danger">
            Logout
          </button>
        </div>
      </header>

      <nav className="dashboard-nav" role="navigation" aria-label="Dashboard navigation">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
          aria-current={activeTab === 'overview' ? 'page' : undefined}
        >
          Overview
        </button>
        <button
          className={activeTab === 'trends' ? 'active' : ''}
          onClick={() => setActiveTab('trends')}
          aria-current={activeTab === 'trends' ? 'page' : undefined}
        >
          Trends
        </button>
        <button
          className={activeTab === 'correlations' ? 'active' : ''}
          onClick={() => setActiveTab('correlations')}
          aria-current={activeTab === 'correlations' ? 'page' : undefined}
        >
          Correlations
        </button>
        <button
          className={activeTab === 'calendar' ? 'active' : ''}
          onClick={() => setActiveTab('calendar')}
          aria-current={activeTab === 'calendar' ? 'page' : undefined}
        >
          Calendar
        </button>
        <button
          className={activeTab === 'alerts' ? 'active' : ''}
          onClick={() => setActiveTab('alerts')}
          aria-current={activeTab === 'alerts' ? 'page' : undefined}
        >
          Alerts {alerts.length > 0 && `(${alerts.length})`}
        </button>
      </nav>

      <div className="date-range-selector">
        <label htmlFor="date-range">Time Range:</label>
        <select
          id="date-range"
          value={dateRange}
          onChange={(e) => setDateRange(Number(e.target.value))}
        >
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
      </div>

      <main id="main-content" className="dashboard-content">
        {activeTab === 'overview' && (
          <div className="overview-grid">
            <StatisticsPanel statistics={statistics} />
            <div className="card">
              <h2>Mood Trend</h2>
              <TrendChart data={moodData} label="Mood Score" color="#3498db" />
            </div>
            <div className="card">
              <h2>Sleep Duration</h2>
              <TrendChart data={sleepData} label="Hours" color="#2ecc71" />
            </div>
            <div className="card">
              <h2>Energy Level</h2>
              <TrendChart data={energyData} label="Energy Score" color="#f39c12" />
            </div>
            <div className="card">
              <h2>Emotion Distribution</h2>
              <EmotionPieChart data={emotionData} />
            </div>
          </div>
        )}

        {activeTab === 'trends' && (
          <div className="trends-grid">
            <div className="card full-width">
              <h2>Mood Trend ({dateRange} days)</h2>
              <TrendChart data={moodData} label="Mood Score" color="#3498db" />
            </div>
            <div className="card full-width">
              <h2>Sleep Duration ({dateRange} days)</h2>
              <TrendChart data={sleepData} label="Hours" color="#2ecc71" />
            </div>
            <div className="card full-width">
              <h2>Energy Level ({dateRange} days)</h2>
              <TrendChart data={energyData} label="Energy Score" color="#f39c12" />
            </div>
            <div className="card full-width">
              <h2>Vital Signs (Last 7 days)</h2>
              <VitalSignsChart data={vitalData} />
            </div>
          </div>
        )}

        {activeTab === 'correlations' && (
          <div className="correlations-grid">
            <div className="card">
              <h2>Sleep vs Mood Correlation</h2>
              <CorrelationChart 
                xData={sleepData} 
                yData={moodData}
                xLabel="Sleep (hours)"
                yLabel="Mood Score"
              />
            </div>
            <div className="card">
              <h2>Energy vs Sleep Correlation</h2>
              <CorrelationChart 
                xData={sleepData} 
                yData={energyData}
                xLabel="Sleep (hours)"
                yLabel="Energy Score"
              />
            </div>
          </div>
        )}

        {activeTab === 'calendar' && (
          <CalendarView 
            moodData={moodData}
            sleepData={sleepData}
            energyData={energyData}
          />
        )}

        {activeTab === 'alerts' && (
          <AlertsPanel 
            alerts={alerts}
            onAcknowledge={handleAcknowledgeAlert}
          />
        )}
      </main>
    </div>
  );
}

export default Dashboard;
