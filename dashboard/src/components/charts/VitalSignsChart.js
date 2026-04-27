import React, { useRef } from 'react';
import { Line } from 'react-chartjs-2';
import html2canvas from 'html2canvas';

function VitalSignsChart({ data }) {
  const chartRef = useRef(null);

  if (!data || !data.vitals || data.vitals.length === 0) {
    return <div className="no-data">No vital signs data available</div>;
  }

  // Group vitals by type
  const vitalsByType = {};
  data.vitals.forEach(vital => {
    if (!vitalsByType[vital.vital_type]) {
      vitalsByType[vital.vital_type] = [];
    }
    vitalsByType[vital.vital_type].push(vital);
  });

  const colors = {
    heart_rate: '#e74c3c',
    spo2: '#3498db',
    temperature: '#f39c12'
  };

  const labels = {
    heart_rate: 'Heart Rate (bpm)',
    spo2: 'SpO2 (%)',
    temperature: 'Temperature (°C)'
  };

  const datasets = Object.keys(vitalsByType).map(type => ({
    label: labels[type] || type,
    data: vitalsByType[type].map(v => ({
      x: new Date(v.timestamp).toLocaleDateString(),
      y: v.value
    })),
    borderColor: colors[type] || '#95a5a6',
    backgroundColor: `${colors[type] || '#95a5a6'}33`,
    tension: 0.4,
    pointRadius: 4,
    pointHoverRadius: 6,
  }));

  const chartData = {
    datasets: datasets
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      }
    },
    scales: {
      y: {
        beginAtZero: false,
        grid: {
          color: '#f0f0f0'
        }
      },
      x: {
        type: 'category',
        grid: {
          display: false
        }
      }
    }
  };

  const handleExportPNG = async () => {
    if (chartRef.current) {
      const canvas = await html2canvas(chartRef.current);
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = 'vital-signs-chart.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="chart-container" ref={chartRef}>
      <div className="chart-controls">
        <button onClick={handleExportPNG} className="btn-small">
          📷 Export PNG
        </button>
      </div>
      <div style={{ height: '300px' }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}

export default VitalSignsChart;
