import { useState, useRef } from 'react';

function E1RMChart({ history = [], predictedE1RM, chainedPredictions = [], exerciseName, targetReps = 8 }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [selectedRepCount, setSelectedRepCount] = useState(targetReps);
  const svgRef = useRef(null);

  // Format date strings like "2026-08-20" -> "Aug 20"
  const formatDateLabel = (dateStr) => {
    if (!dateStr || dateStr.startsWith('Session') || dateStr.startsWith('Next') || dateStr.startsWith('Predicted')) return dateStr;
    try {
      const parts = dateStr.split('-');
      if (parts.length >= 3) {
        const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const mIdx = parseInt(parts[1], 10) - 1;
        const day = parseInt(parts[2], 10);
        if (mIdx >= 0 && mIdx < 12) {
          return `${monthNames[mIdx]} ${day < 10 ? '0' + day : day}`;
        }
      }
      return dateStr;
    } catch {
      return dateStr;
    }
  };

  // Combine historical points + predicted point(s) into a unified dataset
  const historicalPoints = (history && history.length > 0)
    ? history.map((pt, i) => ({
        id: `hist-${i}`,
        label: formatDateLabel(pt.date) || `Session #${i + 1}`,
        rawDate: pt.date,
        e1RM: Number(pt.e1RM),
        weight: pt.weight,
        reps: pt.reps,
        isPrediction: false
      }))
    : [
        { id: 'mock-1', label: 'Jul 15', e1RM: 200, weight: 165, reps: 8, isPrediction: false },
        { id: 'mock-2', label: 'Jul 22', e1RM: 210, weight: 175, reps: 8, isPrediction: false },
        { id: 'mock-3', label: 'Aug 01', e1RM: 215, weight: 180, reps: 8, isPrediction: false },
        { id: 'mock-4', label: 'Aug 10', e1RM: 220, weight: 185, reps: 8, isPrediction: false }
      ];

  const allPoints = [...historicalPoints];
  if (chainedPredictions && chainedPredictions.length > 0) {
    chainedPredictions.forEach((predVal, idx) => {
      const step = idx + 1;
      allPoints.push({
        id: `pred-point-${idx}`,
        label: `+${step} Wk`,
        rawDate: `Predicted Workout (+${step})`,
        e1RM: Number(predVal),
        weight: null,
        reps: null,
        isPrediction: true,
        stepIndex: step
      });
    });
  } else if (predictedE1RM != null) {
    allPoints.push({
      id: 'pred-point-0',
      label: '+1 Wk',
      rawDate: 'Predicted Workout (+1)',
      e1RM: Number(predictedE1RM),
      weight: null,
      reps: null,
      isPrediction: true,
      stepIndex: 1
    });
  }

  // Reversed 1RM calculation: For 1 rep, weight = e1RM. For > 1 rep, weight = e1RM / (1 + reps/30)
  const calculateWeightForReps = (e1rm, reps) => {
    if (!e1rm || e1rm <= 0) return 0;
    if (reps === 1) return e1rm;
    return e1rm / (1.0 + reps / 30.0);
  };

  const activePredictedE1RM = (chainedPredictions && chainedPredictions.length > 0)
    ? Number(chainedPredictions[0])
    : (predictedE1RM != null ? Number(predictedE1RM) : (allPoints.length > 0 ? allPoints[allPoints.length - 1].e1RM : 200));
  const predictedWeightForTarget = calculateWeightForReps(activePredictedE1RM, selectedRepCount);

  // SVG Chart Layout Math
  const width = 760;
  const height = 260;
  const paddingLeft = 65;
  const paddingRight = 50;
  const paddingTop = 30;
  const paddingBottom = 45;

  const minE1RM = Math.floor(Math.min(...allPoints.map(p => p.e1RM)) * 0.92);
  const maxE1RM = Math.ceil(Math.max(...allPoints.map(p => p.e1RM)) * 1.08);
  const e1RMRange = (maxE1RM - minE1RM) || 1;

  const getX = (index) => {
    if (allPoints.length === 1) return (width + paddingLeft) / 2;
    return paddingLeft + (index / (allPoints.length - 1)) * (width - paddingLeft - paddingRight);
  };

  const getY = (val) => {
    return height - paddingBottom - ((val - minE1RM) / e1RMRange) * (height - paddingTop - paddingBottom);
  };

  // Build SVG Path Strings
  const pointCoordinates = allPoints.map((pt, i) => ({
    x: getX(i),
    y: getY(pt.e1RM),
    data: pt
  }));

  const pathD = pointCoordinates.reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
  }, '');

  const areaD = pointCoordinates.length > 0
    ? `${pathD} L ${pointCoordinates[pointCoordinates.length - 1].x} ${height - paddingBottom} L ${pointCoordinates[0].x} ${height - paddingBottom} Z`
    : '';

  // Track closest point on mouse move across entire SVG area
  const handleMouseMove = (e) => {
    if (!svgRef.current || pointCoordinates.length === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const svgX = (mouseX / rect.width) * width;

    let nearest = pointCoordinates[0];
    let minDist = Math.abs(pointCoordinates[0].x - svgX);

    for (let i = 1; i < pointCoordinates.length; i++) {
      const dist = Math.abs(pointCoordinates[i].x - svgX);
      if (dist < minDist) {
        minDist = dist;
        nearest = pointCoordinates[i];
      }
    }
    setHoveredPoint(nearest.data);
  };

  const repPresets = [1, 3, 5, 8, 10, 12, 15];

  // Determine tick display skip count to prevent overlapping date labels
  const totalPoints = allPoints.length;
  const labelSkipInterval = totalPoints > 10 ? Math.ceil(totalPoints / 8) : 1;

  return (
    <div className="chart-container-card">
      {/* Reversed 1RM / Performance Prediction Banner */}
      {predictedE1RM != null && (
        <div className="performance-banner">
          <div className="performance-header">
            <span className="performance-tag">🎯 Predicted Performance Target</span>
            <h3 className="performance-title">
              Recommended Weight: <strong className="weight-highlight">{predictedWeightForTarget.toFixed(1)} lbs</strong> for {selectedRepCount} {selectedRepCount === 1 ? 'rep (1RM)' : 'reps'}
            </h3>
          </div>

          {/* Target Reps Quick-Select Buttons */}
          <div className="rep-selector-group">
            <span className="rep-selector-label">Target Reps:</span>
            {repPresets.map(r => (
              <button
                key={r}
                type="button"
                className={`rep-chip ${selectedRepCount === r ? 'active-chip' : ''}`}
                onClick={() => setSelectedRepCount(r)}
              >
                {r} {r === 1 ? 'rep (1RM)' : 'reps'}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chart Header */}
      <div className="chart-header">
        <div>
          <h4 className="chart-title">1-Rep Max (e1RM) Progression & Prediction</h4>
          <span className="chart-subtitle">{exerciseName || 'Exercise Trajectory'}</span>
        </div>

        <div className="chart-legend">
          <span className="legend-item">
            <span className="legend-dot history-dot"></span> History
          </span>
          <span className="legend-item">
            <span className="legend-dot prediction-dot"></span> Prediction
          </span>
        </div>
      </div>

      {/* SVG Line Graph with Mouse Proximity Tracking */}
      <div className="svg-wrapper">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="chart-svg"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredPoint(null)}
        >
          <defs>
            {/* Area Fill Gradient */}
            <linearGradient id="chartAreaGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a855f7" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
            </linearGradient>

            {/* Stroke Line Gradient */}
            <linearGradient id="chartLineGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#a855f7" />
              <stop offset="80%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#ec4899" />
            </linearGradient>

            {/* Neon Glow Filter */}
            <filter id="predictionGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Y-Axis Title */}
          <text
            x={14}
            y={height / 2}
            transform={`rotate(-90 14 ${height / 2})`}
            className="grid-text-y"
            style={{ textAnchor: 'middle', fontWeight: 600, fill: 'rgba(255, 255, 255, 0.45)', fontSize: '11px' }}
          >
            1RM Weight (lbs)
          </text>

          {/* Grid Y Lines */}
          {[0, 0.33, 0.66, 1].map((pct, idx) => {
            const val = minE1RM + pct * e1RMRange;
            const y = getY(val);
            return (
              <g key={idx}>
                <line x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} className="grid-line" />
                <text x={paddingLeft - 8} y={y + 4} className="grid-text-y">{Math.round(val)}</text>
              </g>
            );
          })}

          {/* Gradient Area */}
          <path d={areaD} fill="url(#chartAreaGradient)" />

          {/* Connection Line */}
          <path d={pathD} fill="none" stroke="url(#chartLineGradient)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />

          {/* Hover Guidance Line */}
          {hoveredPoint && (
            (() => {
              const activePt = pointCoordinates.find(p => p.data.id === hoveredPoint.id);
              if (!activePt) return null;
              return (
                <line
                  x1={activePt.x}
                  y1={paddingTop}
                  x2={activePt.x}
                  y2={height - paddingBottom}
                  stroke="rgba(255, 255, 255, 0.2)"
                  strokeDasharray="3 3"
                />
              );
            })()
          )}

          {/* Data Points */}
          {pointCoordinates.map((pt, idx) => {
            const isPred = pt.data.isPrediction;
            const isHovered = hoveredPoint && hoveredPoint.id === pt.data.id;
            const showLabel = isPred || idx === 0 || idx === pointCoordinates.length - 1 || idx % labelSkipInterval === 0;

            return (
              <g key={pt.data.id}>
                {isPred ? (
                  /* Glowing Prediction Marker */
                  <g filter="url(#predictionGlow)">
                    <circle cx={pt.x} cy={pt.y} r={isHovered ? 12 : 9} fill="#ec4899" opacity="0.5" />
                    <circle cx={pt.x} cy={pt.y} r={isHovered ? 8 : 6} fill="#ec4899" stroke="#ffffff" strokeWidth="2" />
                    <circle cx={pt.x} cy={pt.y} r="3" fill="#ffffff" />
                  </g>
                ) : (
                  /* Historical Data Marker */
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={isHovered ? 7 : 4.5}
                    fill={isHovered ? '#3b82f6' : '#a855f7'}
                    stroke="#ffffff"
                    strokeWidth={isHovered ? 2 : 1.5}
                    className="history-node"
                  />
                )}

                {/* X Date Labels (spaced cleanly without overlapping) */}
                {showLabel && (
                  <text
                    x={pt.x}
                    y={height - 14}
                    className={`grid-text-x ${isPred ? 'pred-x-label' : ''}`}
                  >
                    {pt.data.label}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Hover Tooltip Popup */}
        {hoveredPoint && (
          <div className="chart-tooltip">
            <div className="tooltip-title">{hoveredPoint.rawDate || hoveredPoint.label}</div>
            <div className="tooltip-metric">
              e1RM: <strong>{hoveredPoint.e1RM.toFixed(1)} lbs</strong>
            </div>
            {Boolean(hoveredPoint.weight && hoveredPoint.weight > 0) && (
              <div className="tooltip-sub">
                Worked: {hoveredPoint.weight} lbs × {hoveredPoint.reps} reps
              </div>
            )}
            {hoveredPoint.isPrediction && (
              <div className="tooltip-sub prediction-sub">
                ⚡ Model Predicted Target {hoveredPoint.stepIndex ? `(+${hoveredPoint.stepIndex} Workout)` : ''}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default E1RMChart;
