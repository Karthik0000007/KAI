import React, { useState } from 'react';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import './CalendarView.css';

function CalendarView({ moodData, sleepData, energyData }) {
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [view, setView] = useState('month');

  // Create a map of dates to health data
  const healthDataMap = {};
  
  if (moodData && moodData.data) {
    moodData.data.forEach(d => {
      const date = d.date.split('T')[0];
      if (!healthDataMap[date]) healthDataMap[date] = {};
      healthDataMap[date].mood = d.value;
    });
  }

  if (sleepData && sleepData.data) {
    sleepData.data.forEach(d => {
      const date = d.date.split('T')[0];
      if (!healthDataMap[date]) healthDataMap[date] = {};
      healthDataMap[date].sleep = d.value;
    });
  }

  if (energyData && energyData.data) {
    energyData.data.forEach(d => {
      const date = d.date.split('T')[0];
      if (!healthDataMap[date]) healthDataMap[date] = {};
      healthDataMap[date].energy = d.value;
    });
  }

  const getTileContent = ({ date, view }) => {
    if (view === 'month') {
      const dateStr = date.toISOString().split('T')[0];
      const data = healthDataMap[dateStr];
      
      if (data) {
        return (
          <div className="calendar-tile-content">
            {data.mood !== undefined && (
              <div className="tile-indicator mood" title={`Mood: ${data.mood.toFixed(1)}`}>
                😊
              </div>
            )}
            {data.sleep !== undefined && (
              <div className="tile-indicator sleep" title={`Sleep: ${data.sleep.toFixed(1)}h`}>
                😴
              </div>
            )}
            {data.energy !== undefined && (
              <div className="tile-indicator energy" title={`Energy: ${data.energy.toFixed(1)}`}>
                ⚡
              </div>
            )}
          </div>
        );
      }
    }
    return null;
  };

  const getTileClassName = ({ date, view }) => {
    if (view === 'month') {
      const dateStr = date.toISOString().split('T')[0];
      const data = healthDataMap[dateStr];
      
      if (data) {
        if (data.mood !== undefined && data.mood < 3) {
          return 'low-mood-day';
        }
        if (data.sleep !== undefined && data.sleep < 6) {
          return 'low-sleep-day';
        }
      }
    }
    return null;
  };

  const getSelectedDateData = () => {
    const dateStr = selectedDate.toISOString().split('T')[0];
    return healthDataMap[dateStr] || null;
  };

  const selectedData = getSelectedDateData();

  return (
    <div className="calendar-view">
      <div className="card">
        <h2>Health Calendar</h2>
        <p className="calendar-description">
          View your daily health metrics. Days with low mood or insufficient sleep are highlighted.
        </p>
        
        <Calendar
          onChange={setSelectedDate}
          value={selectedDate}
          tileContent={getTileContent}
          tileClassName={getTileClassName}
          className="health-calendar"
        />

        <div className="calendar-legend">
          <div className="legend-item">
            <span className="legend-icon">😊</span> Mood tracked
          </div>
          <div className="legend-item">
            <span className="legend-icon">😴</span> Sleep tracked
          </div>
          <div className="legend-item">
            <span className="legend-icon">⚡</span> Energy tracked
          </div>
          <div className="legend-item">
            <span className="legend-color low-mood"></span> Low mood day
          </div>
          <div className="legend-item">
            <span className="legend-color low-sleep"></span> Insufficient sleep
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Selected Date: {selectedDate.toLocaleDateString()}</h2>
        {selectedData ? (
          <div className="date-details">
            {selectedData.mood !== undefined && (
              <div className="detail-item">
                <span className="detail-label">Mood Score:</span>
                <span className="detail-value">{selectedData.mood.toFixed(1)} / 5</span>
                <div className="detail-bar">
                  <div 
                    className="detail-bar-fill mood-bar" 
                    style={{ width: `${(selectedData.mood / 5) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
            {selectedData.sleep !== undefined && (
              <div className="detail-item">
                <span className="detail-label">Sleep Duration:</span>
                <span className="detail-value">{selectedData.sleep.toFixed(1)} hours</span>
                <div className="detail-bar">
                  <div 
                    className="detail-bar-fill sleep-bar" 
                    style={{ width: `${Math.min((selectedData.sleep / 10) * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
            )}
            {selectedData.energy !== undefined && (
              <div className="detail-item">
                <span className="detail-label">Energy Level:</span>
                <span className="detail-value">{selectedData.energy.toFixed(1)} / 5</span>
                <div className="detail-bar">
                  <div 
                    className="detail-bar-fill energy-bar" 
                    style={{ width: `${(selectedData.energy / 5) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="no-data">No health data recorded for this date.</p>
        )}
      </div>
    </div>
  );
}

export default CalendarView;
