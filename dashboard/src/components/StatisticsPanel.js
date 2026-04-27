import React from 'react';
import './StatisticsPanel.css';

function StatisticsPanel({ statistics }) {
  if (!statistics) {
    return <div className="no-data">No statistics available</div>;
  }

  const stats = [
    {
      label: 'Total Check-ins',
      value: statistics.total_checkins || 0,
      icon: '📝',
      color: '#3498db'
    },
    {
      label: 'Average Mood',
      value: statistics.avg_mood ? statistics.avg_mood.toFixed(1) : 'N/A',
      icon: '😊',
      color: '#f39c12',
      suffix: '/5'
    },
    {
      label: 'Average Sleep',
      value: statistics.avg_sleep ? statistics.avg_sleep.toFixed(1) : 'N/A',
      icon: '😴',
      color: '#2ecc71',
      suffix: 'h'
    },
    {
      label: 'Average Energy',
      value: statistics.avg_energy ? statistics.avg_energy.toFixed(1) : 'N/A',
      icon: '⚡',
      color: '#e67e22',
      suffix: '/5'
    },
    {
      label: 'Active Alerts',
      value: statistics.active_alerts || 0,
      icon: '🔔',
      color: '#e74c3c'
    },
    {
      label: 'Conversations',
      value: statistics.total_conversations || 0,
      icon: '💬',
      color: '#9b59b6'
    }
  ];

  return (
    <div className="statistics-panel card">
      <h2>Health Statistics</h2>
      <div className="stats-grid">
        {stats.map((stat, index) => (
          <div key={index} className="stat-card" style={{ borderLeftColor: stat.color }}>
            <div className="stat-icon" style={{ color: stat.color }}>
              {stat.icon}
            </div>
            <div className="stat-content">
              <div className="stat-value">
                {stat.value}
                {stat.suffix && <span className="stat-suffix">{stat.suffix}</span>}
              </div>
              <div className="stat-label">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>

      {statistics.streak && statistics.streak > 0 && (
        <div className="streak-banner">
          <span className="streak-icon">🔥</span>
          <span className="streak-text">
            {statistics.streak} day streak! Keep it up!
          </span>
        </div>
      )}
    </div>
  );
}

export default StatisticsPanel;
