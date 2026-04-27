import React, { useState } from 'react';
import './AlertsPanel.css';

function AlertsPanel({ alerts, onAcknowledge }) {
  const [filter, setFilter] = useState('all'); // all, unacknowledged, acknowledged

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'unacknowledged') return !alert.acknowledged;
    if (filter === 'acknowledged') return alert.acknowledged;
    return true;
  });

  const getSeverityClass = (severity) => {
    return `severity-${severity.toLowerCase()}`;
  };

  const getSeverityIcon = (severity) => {
    switch (severity.toLowerCase()) {
      case 'high': return '🔴';
      case 'medium': return '🟡';
      case 'low': return '🟢';
      default: return '⚪';
    }
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  return (
    <div className="alerts-panel">
      <div className="card">
        <div className="alerts-header">
          <h2>Proactive Health Alerts</h2>
          <div className="alerts-filter">
            <button
              className={filter === 'all' ? 'active' : ''}
              onClick={() => setFilter('all')}
            >
              All ({alerts.length})
            </button>
            <button
              className={filter === 'unacknowledged' ? 'active' : ''}
              onClick={() => setFilter('unacknowledged')}
            >
              Unacknowledged ({alerts.filter(a => !a.acknowledged).length})
            </button>
            <button
              className={filter === 'acknowledged' ? 'active' : ''}
              onClick={() => setFilter('acknowledged')}
            >
              Acknowledged ({alerts.filter(a => a.acknowledged).length})
            </button>
          </div>
        </div>

        {filteredAlerts.length === 0 ? (
          <div className="no-alerts">
            <p>✅ No alerts to display</p>
            <p className="no-alerts-subtitle">
              {filter === 'all' 
                ? "You're doing great! Keep up the healthy habits."
                : `No ${filter} alerts.`
              }
            </p>
          </div>
        ) : (
          <div className="alerts-list">
            {filteredAlerts.map(alert => (
              <div 
                key={alert.alert_id} 
                className={`alert-item ${getSeverityClass(alert.severity)} ${alert.acknowledged ? 'acknowledged' : ''}`}
                role="article"
                aria-label={`${alert.severity} severity alert: ${alert.message}`}
              >
                <div className="alert-icon">
                  {getSeverityIcon(alert.severity)}
                </div>
                <div className="alert-content">
                  <div className="alert-header-row">
                    <h3 className="alert-title">{alert.pattern_type.replace(/_/g, ' ')}</h3>
                    <span className={`alert-badge ${getSeverityClass(alert.severity)}`}>
                      {alert.severity}
                    </span>
                  </div>
                  <p className="alert-message">{alert.message}</p>
                  {alert.explanation && (
                    <p className="alert-explanation">{alert.explanation}</p>
                  )}
                  {alert.context && Object.keys(alert.context).length > 0 && (
                    <div className="alert-context">
                      <strong>Details:</strong>
                      <ul>
                        {Object.entries(alert.context).map(([key, value]) => (
                          <li key={key}>
                            {key.replace(/_/g, ' ')}: {typeof value === 'number' ? value.toFixed(1) : value}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="alert-footer">
                    <span className="alert-date">{formatDate(alert.created_at)}</span>
                    {!alert.acknowledged ? (
                      <button
                        className="btn-acknowledge"
                        onClick={() => onAcknowledge(alert.alert_id)}
                        aria-label={`Acknowledge alert: ${alert.message}`}
                      >
                        ✓ Acknowledge
                      </button>
                    ) : (
                      <span className="acknowledged-badge">
                        ✓ Acknowledged {alert.acknowledged_at && `on ${formatDate(alert.acknowledged_at)}`}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default AlertsPanel;
