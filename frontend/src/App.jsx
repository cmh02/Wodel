import { useState, useEffect } from 'react';
import './App.css';

const API_HOST = "http://localhost:8000";

function App() {
  // File upload state
  const [workoutFile, setWorkoutFile] = useState(null);
  const [biometricsFile, setBiometricsFile] = useState(null);
  const [userBirthday, setUserBirthday] = useState("2002-05-14");
  const [userCurrentAge, setUserCurrentAge] = useState(24);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainingError, setTrainingError] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);

  // Prediction Mode State: 'simple' | 'advanced'
  const [predictionMode, setPredictionMode] = useState('simple');

  // Simple Mode Prediction Inputs
  const [simpleInputs, setSimpleInputs] = useState({
    Name: "Romanian Deadlift (Barbell)",
    Set_Order: "3",
    Reps: 8,
    Weight: 185.0,
    exerciseOrderInWorkout: 2,
    birthday: "2002-05-14",
    current_age: 24
  });

  // Advanced Mode Prediction Inputs
  const [predictionInputs, setPredictionInputs] = useState({
    Name: "Romanian Deadlift (Barbell)",
    Set_Order: "3",
    Distance: 0.0,
    e1RMLag1: 225.0,
    e1RMLag2: 220.0,
    e1RMLag3: 215.0,
    timeSinceLastWorkout: 2.0,
    timeSinceLastSameExercise: 7.0,
    exerciseOrderInWorkout: 2,
    Age: 24.0,
    Body_Weight: 140.0,
    BMI: 23.0,
    Body_Fat: 15.0,
    Fat_Free_Mass: 119.0,
    Subcutaneous_Fat: 13.0,
    Visceral_Fat: 6.0,
    Body_Water: 61.0,
    Skeletal_Muscle: 55.0,
    Muscle_Mass: 113.0,
    Bone_Mass: 6.0,
    Protein: 19.0,
    BMR: 1540.0,
    Metabolic_Age: 18.0
  });

  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  // Common exercises list
  const commonExercises = [
    "Romanian Deadlift (Barbell)",
    "Bench Press (Dumbbell)",
    "Chest Press (Machine)",
    "Overhead Press (Barbell)",
    "Squat (Smith Machine)",
    "Bicep Curl (Dumbbell)",
    "Lying Leg Curl (Machine)",
    "Lateral Raise (Dumbbell)",
    "Face Pull (Cable)",
    "Leg Press",
    "Hack Squat",
    "Seated Leg Press (Machine)",
    "Standing Calf Raise (Machine)",
    "Bayesian Cable Curl",
    "Back Extension",
    "Reverse Fly (Dumbbell)"
  ];

  const handleBirthdayChange = (newVal) => {
    setUserBirthday(newVal);
    setSimpleInputs(prev => ({ ...prev, birthday: newVal }));
  };

  const handleAgeChange = (newVal) => {
    setUserCurrentAge(newVal);
    setSimpleInputs(prev => ({ ...prev, current_age: newVal }));
    setPredictionInputs(prev => ({ ...prev, Age: parseFloat(newVal) || 24.0 }));
  };

  // Check model status on mount
  useEffect(() => {
    fetch(`${API_HOST}/api/status`)
      .then(res => res.json())
      .then(data => {
        if (data.modelTrained) {
          setTrainingResult({
            bestModel: data.bestModel,
            bestR2: data.bestR2,
            isFromStatus: true
          });
        }
      })
      .catch(err => console.error("Error fetching model status:", err));
  }, []);

  const handleTrainSubmit = async (e) => {
    e.preventDefault();
    if (!workoutFile || !biometricsFile) {
      setTrainingError("Please select both a workout logs file and a biometrics file.");
      return;
    }

    setTrainingLoading(true);
    setTrainingError(null);
    setTrainingResult(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append("workoutFile", workoutFile);
    formData.append("biometricsFile", biometricsFile);
    if (userBirthday) formData.append("birthday", userBirthday);
    if (userCurrentAge) formData.append("currentAge", userCurrentAge);

    try {
      const response = await fetch(`${API_HOST}/api/train`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Training failed.");
      }

      const data = await response.json();
      setTrainingResult(data);
    } catch (err) {
      setTrainingError(err.message);
    } finally {
      setTrainingLoading(false);
    }
  };

  const handlePredictSubmit = async (e) => {
    e.preventDefault();
    setPredictLoading(true);
    setPredictError(null);
    setPredictionResult(null);

    const endpoint = predictionMode === 'simple' ? '/api/predict/simple' : '/api/predict';
    const payload = predictionMode === 'simple'
      ? {
          ...simpleInputs,
          birthday: userBirthday,
          current_age: parseFloat(userCurrentAge) || 24.0
        }
      : predictionInputs;

    try {
      const response = await fetch(`${API_HOST}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed.");
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setPredictError(err.message);
    } finally {
      setPredictLoading(false);
    }
  };

  const handleSimpleInputChange = (fieldName, value) => {
    setSimpleInputs(prev => ({
      ...prev,
      [fieldName]: value
    }));
  };

  const handleAdvancedInputChange = (fieldName, value) => {
    setPredictionInputs(prev => ({
      ...prev,
      [fieldName]: value
    }));
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="gradient-title" id="main-title">Wodle AI Dashboard</h1>
        <p className="subtitle">Load workout data and predict Estimated 1-Rep Max (e1RM)</p>
      </header>

      <div className="dashboard-grid">
        {/* Step 1: File Loading & Model Training */}
        <section className="glass-card">
          <h2 className="form-section-title">1. Train Best Predictive Model</h2>
          <form onSubmit={handleTrainSubmit}>
            <div className="upload-grid">
              {/* Workout File Box */}
              <label className="upload-box" id="workout-upload-label">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setWorkoutFile(e.target.files[0])}
                  style={{ display: 'none' }}
                  id="workout-file-input"
                />
                <span className="upload-icon">🏋️‍♂️</span>
                <p><strong>Workout Logs CSV</strong></p>
                {workoutFile ? (
                  <span className="file-name">{workoutFile.name}</span>
                ) : (
                  <span>Select strong_workouts.csv</span>
                )}
              </label>

              {/* Biometrics File Box */}
              <label className="upload-box" id="biometrics-upload-label">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setBiometricsFile(e.target.files[0])}
                  style={{ display: 'none' }}
                  id="biometrics-file-input"
                />
                <span className="upload-icon">⚖️</span>
                <p><strong>Renpho Biometrics CSV</strong></p>
                {biometricsFile ? (
                  <span className="file-name">{biometricsFile.name}</span>
                ) : (
                  <span>Select renpho.csv</span>
                )}
              </label>
            </div>

            {/* Optional Personal Demographics */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
              <div className="input-group">
                <label htmlFor="user-birthday">User Birthday</label>
                <input
                  type="date"
                  id="user-birthday"
                  className="input-control"
                  value={userBirthday}
                  onChange={(e) => handleBirthdayChange(e.target.value)}
                />
              </div>
              <div className="input-group">
                <label htmlFor="user-age">Current Age (Years)</label>
                <input
                  type="number"
                  step="0.1"
                  id="user-age"
                  className="input-control"
                  value={userCurrentAge}
                  onChange={(e) => handleAgeChange(e.target.value)}
                />
              </div>
            </div>

            {trainingError && <div style={{ color: '#ef4444', marginBottom: '15px', fontWeight: '500' }}>{trainingError}</div>}

            <button
              type="submit"
              className="btn-primary"
              disabled={trainingLoading || !workoutFile || !biometricsFile}
              id="train-btn"
            >
              {trainingLoading ? (
                <>
                  <div className="spinner"></div>
                  Preprocessing and Training Models...
                </>
              ) : (
                "Train & Find Best Model"
              )}
            </button>
          </form>

          {/* Training Results */}
          {trainingResult && (
            <div className="metrics-section" style={{ marginTop: '30px' }}>
              <div className="best-model-badge" id="best-model-badge">
                🎉 Best Model: <strong>{trainingResult.bestModel}</strong> 
                {trainingResult.bestR2 && ` (R²: ${trainingResult.bestR2.toFixed(4)})`}
              </div>

              {trainingResult.metrics && (
                <div className="metrics-table-wrapper">
                  <table className="metrics-table">
                    <thead>
                      <tr>
                        <th>Model Name</th>
                        <th>R² Score</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>MAPE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(trainingResult.metrics).map(([name, metrics]) => {
                        const isBest = name === trainingResult.bestModel;
                        return (
                          <tr key={name} className={isBest ? 'highlighted-row' : ''}>
                            <td><strong>{name}</strong> {isBest && "⭐️"}</td>
                            <td className="metric-value">{metrics.r2.toFixed(4)}</td>
                            <td className="metric-value">{metrics.rmse.toFixed(4)}</td>
                            <td className="metric-value">{metrics.mae.toFixed(2)} lbs</td>
                            <td className="metric-value">{(metrics.mape * 100.0).toFixed(2)}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Step 2: Make Predictions */}
        <section className="glass-card">
          <h2 className="form-section-title">2. Predict Estimated 1-Rep Max (e1RM)</h2>

          {/* Mode Switcher Tabs */}
          <div className="mode-toggle-group">
            <button
              type="button"
              className={`mode-tab ${predictionMode === 'simple' ? 'active-mode' : ''}`}
              onClick={() => { setPredictionMode('simple'); setPredictionResult(null); }}
              id="simple-mode-tab"
            >
              ⚡️ Simple Prediction Mode
            </button>
            <button
              type="button"
              className={`mode-tab ${predictionMode === 'advanced' ? 'active-mode' : ''}`}
              onClick={() => { setPredictionMode('advanced'); setPredictionResult(null); }}
              id="advanced-mode-tab"
            >
              ⚙️ Advanced Prediction Mode
            </button>
          </div>

          <form onSubmit={handlePredictSubmit}>
            {predictionMode === 'simple' ? (
              /* --- SIMPLE PREDICTION FORM --- */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  <div className="input-group">
                    <label htmlFor="simple-birthday">Birthday</label>
                    <input
                      type="date"
                      id="simple-birthday"
                      className="input-control"
                      value={userBirthday}
                      onChange={(e) => handleBirthdayChange(e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label htmlFor="simple-age">Current Age</label>
                    <input
                      type="number"
                      step="0.1"
                      id="simple-age"
                      className="input-control"
                      value={userCurrentAge}
                      onChange={(e) => handleAgeChange(e.target.value)}
                    />
                  </div>
                </div>

                <div className="input-group">
                  <label htmlFor="simple-name">Exercise Name</label>
                  <select
                    id="simple-name"
                    className="input-control"
                    value={simpleInputs.Name}
                    onChange={(e) => handleSimpleInputChange("Name", e.target.value)}
                  >
                    {commonExercises.map(ex => (
                      <option key={ex} value={ex}>{ex}</option>
                    ))}
                  </select>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  {/* Set Order Slider */}
                  <div className="slider-container">
                    <div className="slider-header">
                      <label htmlFor="simple-set-order">Set Order</label>
                      <span className="slider-badge">Set {simpleInputs.Set_Order}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      id="simple-set-order"
                      className="range-input"
                      value={simpleInputs.Set_Order}
                      onChange={(e) => handleSimpleInputChange("Set_Order", e.target.value)}
                    />
                  </div>

                  {/* Exercise Order Slider */}
                  <div className="slider-container">
                    <div className="slider-header">
                      <label htmlFor="simple-ex-order">Exercise Order in Workout</label>
                      <span className="slider-badge">#{simpleInputs.exerciseOrderInWorkout}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      id="simple-ex-order"
                      className="range-input"
                      value={simpleInputs.exerciseOrderInWorkout}
                      onChange={(e) => handleSimpleInputChange("exerciseOrderInWorkout", parseInt(e.target.value) || 1)}
                    />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  <div className="input-group">
                    <label htmlFor="simple-reps"># of Reps</label>
                    <input
                      type="number"
                      min="1"
                      max="50"
                      id="simple-reps"
                      className="input-control"
                      value={simpleInputs.Reps}
                      onChange={(e) => handleSimpleInputChange("Reps", parseInt(e.target.value) || 1)}
                    />
                  </div>
                  <div className="input-group">
                    <label htmlFor="simple-weight">Weight (lbs)</label>
                    <input
                      type="number"
                      step="0.5"
                      id="simple-weight"
                      className="input-control"
                      value={simpleInputs.Weight}
                      onChange={(e) => handleSimpleInputChange("Weight", parseFloat(e.target.value) || 0)}
                    />
                  </div>
                </div>
              </div>
            ) : (
              /* --- ADVANCED PREDICTION FORM --- */
              <div className="form-columns">
                {/* Column 1: Exercise Info */}
                <div className="form-column">
                  <h3 className="form-column-title">Exercise Info</h3>
                  
                  <div className="input-group">
                    <label htmlFor="input-name">Exercise Name</label>
                    <select
                      id="input-name"
                      className="input-control"
                      value={predictionInputs.Name}
                      onChange={(e) => handleAdvancedInputChange("Name", e.target.value)}
                    >
                      {commonExercises.map(ex => (
                        <option key={ex} value={ex}>{ex}</option>
                      ))}
                    </select>
                  </div>

                  {/* Set Order Slider */}
                  <div className="slider-container">
                    <div className="slider-header">
                      <label htmlFor="input-set-order">Set Order</label>
                      <span className="slider-badge">Set {predictionInputs.Set_Order}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      id="input-set-order"
                      className="range-input"
                      value={predictionInputs.Set_Order}
                      onChange={(e) => handleAdvancedInputChange("Set_Order", e.target.value)}
                    />
                  </div>

                  <div className="input-group">
                    <label htmlFor="input-distance">Distance (miles)</label>
                    <input
                      type="number"
                      step="0.01"
                      id="input-distance"
                      className="input-control"
                      value={predictionInputs.Distance}
                      onChange={(e) => handleAdvancedInputChange("Distance", parseFloat(e.target.value) || 0)}
                    />
                  </div>

                  {/* Exercise Order Slider */}
                  <div className="slider-container">
                    <div className="slider-header">
                      <label htmlFor="input-exercise-order">Exercise Order in Workout</label>
                      <span className="slider-badge">#{predictionInputs.exerciseOrderInWorkout}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      id="input-exercise-order"
                      className="range-input"
                      value={predictionInputs.exerciseOrderInWorkout}
                      onChange={(e) => handleAdvancedInputChange("exerciseOrderInWorkout", parseInt(e.target.value) || 1)}
                    />
                  </div>

                  <div className="input-group">
                    <label htmlFor="input-age-advanced">User Age (years)</label>
                    <input
                      type="number"
                      step="0.1"
                      id="input-age-advanced"
                      className="input-control"
                      value={predictionInputs.Age}
                      onChange={(e) => handleAdvancedInputChange("Age", parseFloat(e.target.value) || 0)}
                    />
                  </div>
                </div>

                {/* Column 2: Historical Lags */}
                <div className="form-column">
                  <h3 className="form-column-title">Performance History</h3>

                  <div className="input-group">
                    <label htmlFor="input-lag1">Lag 1 max e1RM (lbs)</label>
                    <input
                      type="number"
                      step="0.1"
                      id="input-lag1"
                      className="input-control"
                      value={predictionInputs.e1RMLag1}
                      onChange={(e) => handleAdvancedInputChange("e1RMLag1", parseFloat(e.target.value) || 0)}
                    />
                  </div>

                  <div className="input-group">
                    <label htmlFor="input-lag2">Lag 2 max e1RM (lbs)</label>
                    <input
                      type="number"
                      step="0.1"
                      id="input-lag2"
                      className="input-control"
                      value={predictionInputs.e1RMLag2}
                      onChange={(e) => handleAdvancedInputChange("e1RMLag2", parseFloat(e.target.value) || 0)}
                    />
                  </div>

                  <div className="input-group">
                    <label htmlFor="input-lag3">Lag 3 max e1RM (lbs)</label>
                    <input
                      type="number"
                      step="0.1"
                      id="input-lag3"
                      className="input-control"
                      value={predictionInputs.e1RMLag3}
                      onChange={(e) => handleAdvancedInputChange("e1RMLag3", parseFloat(e.target.value) || 0)}
                    />
                  </div>

                  {/* Days Since Last Workout Slider */}
                  <div className="slider-container">
                    <div className="slider-header">
                      <label htmlFor="input-time-last-workout">Days Since Last Workout</label>
                      <span className="slider-badge">{predictionInputs.timeSinceLastWorkout} days</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="30"
                      step="0.5"
                      id="input-time-last-workout"
                      className="range-input"
                      value={predictionInputs.timeSinceLastWorkout}
                      onChange={(e) => handleAdvancedInputChange("timeSinceLastWorkout", parseFloat(e.target.value) || 0)}
                    />
                  </div>

                  {/* Days Since Last Same Exercise Slider */}
                  <div className="slider-container">
                    <div className="slider-header">
                      <label htmlFor="input-time-same-exercise">Days Since Last Same Exercise</label>
                      <span className="slider-badge">{predictionInputs.timeSinceLastSameExercise} days</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="30"
                      step="0.5"
                      id="input-time-same-exercise"
                      className="range-input"
                      value={predictionInputs.timeSinceLastSameExercise}
                      onChange={(e) => handleAdvancedInputChange("timeSinceLastSameExercise", parseFloat(e.target.value) || 0)}
                    />
                  </div>
                </div>

                {/* Column 3: Biometrics */}
                <div className="form-column">
                  <h3 className="form-column-title">Body Biometrics</h3>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div className="input-group">
                      <label htmlFor="input-weight">Weight (lbs)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-weight"
                        className="input-control"
                        value={predictionInputs.Body_Weight}
                        onChange={(e) => handleAdvancedInputChange("Body_Weight", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="input-group">
                      <label htmlFor="input-bmi">BMI</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-bmi"
                        className="input-control"
                        value={predictionInputs.BMI}
                        onChange={(e) => handleAdvancedInputChange("BMI", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div className="input-group">
                      <label htmlFor="input-fat">Body Fat (%)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-fat"
                        className="input-control"
                        value={predictionInputs.Body_Fat}
                        onChange={(e) => handleAdvancedInputChange("Body_Fat", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="input-group">
                      <label htmlFor="input-fat-free">Fat-Free Mass (lb)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-fat-free"
                        className="input-control"
                        value={predictionInputs.Fat_Free_Mass}
                        onChange={(e) => handleAdvancedInputChange("Fat_Free_Mass", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div className="input-group">
                      <label htmlFor="input-subcutaneous">Subcutaneous Fat (%)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-subcutaneous"
                        className="input-control"
                        value={predictionInputs.Subcutaneous_Fat}
                        onChange={(e) => handleAdvancedInputChange("Subcutaneous_Fat", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="input-group">
                      <label htmlFor="input-visceral">Visceral Fat</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-visceral"
                        className="input-control"
                        value={predictionInputs.Visceral_Fat}
                        onChange={(e) => handleAdvancedInputChange("Visceral_Fat", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div className="input-group">
                      <label htmlFor="input-water">Body Water (%)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-water"
                        className="input-control"
                        value={predictionInputs.Body_Water}
                        onChange={(e) => handleAdvancedInputChange("Body_Water", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="input-group">
                      <label htmlFor="input-muscle">Skeletal Muscle (%)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-muscle"
                        className="input-control"
                        value={predictionInputs.Skeletal_Muscle}
                        onChange={(e) => handleAdvancedInputChange("Skeletal_Muscle", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div className="input-group">
                      <label htmlFor="input-muscle-mass">Muscle Mass (lb)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-muscle-mass"
                        className="input-control"
                        value={predictionInputs.Muscle_Mass}
                        onChange={(e) => handleAdvancedInputChange("Muscle_Mass", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="input-group">
                      <label htmlFor="input-bone-mass">Bone Mass (lb)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-bone-mass"
                        className="input-control"
                        value={predictionInputs.Bone_Mass}
                        onChange={(e) => handleAdvancedInputChange("Bone_Mass", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div className="input-group">
                      <label htmlFor="input-protein">Protein (%)</label>
                      <input
                        type="number"
                        step="0.1"
                        id="input-protein"
                        className="input-control"
                        value={predictionInputs.Protein}
                        onChange={(e) => handleAdvancedInputChange("Protein", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div className="input-group">
                      <label htmlFor="input-bmr">BMR (kcal)</label>
                      <input
                        type="number"
                        id="input-bmr"
                        className="input-control"
                        value={predictionInputs.BMR}
                        onChange={(e) => handleAdvancedInputChange("BMR", parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div className="input-group">
                    <label htmlFor="input-metabolic-age">Metabolic Age (years)</label>
                    <input
                      type="number"
                      id="input-metabolic-age"
                      className="input-control"
                      value={predictionInputs.Metabolic_Age}
                      onChange={(e) => handleAdvancedInputChange("Metabolic_Age", parseFloat(e.target.value) || 0)}
                    />
                  </div>
                </div>
              </div>
            )}

            {predictError && <div style={{ color: '#ef4444', margin: '15px 0', fontWeight: '500' }}>{predictError}</div>}

            <button
              type="submit"
              className="btn-primary"
              style={{ marginTop: '20px' }}
              disabled={predictLoading || !trainingResult}
              id="predict-btn"
            >
              {predictLoading ? (
                <>
                  <div className="spinner"></div>
                  Calculating Prediction...
                </>
              ) : (
                `Predict e1RM (${predictionMode === 'simple' ? 'Simple' : 'Advanced'})`
              )}
            </button>
          </form>

          {/* Prediction Result Display */}
          {predictionResult && (
            <div className="glass-card prediction-card" id="prediction-result-card">
              <div className="prediction-result-wrapper">
                <span className="prediction-label">Predicted e1RM</span>
                <span className="prediction-value" id="prediction-value">
                  {predictionResult.prediction.toFixed(2)} <span style={{ fontSize: '2rem' }}>lbs</span>
                </span>
                <div className="prediction-model-details">
                  Target: <strong>{predictionResult.target}</strong> | Model Used: <span>{predictionResult.modelUsed}</span>
                </div>
              </div>

              {/* Simple Mode Auto-Derived Features Card */}
              {predictionResult.derivedFeatures && (
                <div className="derived-card">
                  <h4 className="derived-title">✨ Auto-Derived & Historical Lookups Used</h4>
                  <div className="derived-grid">
                    <div className="derived-item">
                      <span className="derived-label">Current Set e1RM</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.currentSete1RM.toFixed(1)} lbs</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">Calculated Age</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.calculatedAge.toFixed(1)} yrs</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">Lag 1 e1RM</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.e1RMLag1.toFixed(1)} lbs</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">Lag 2 e1RM</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.e1RMLag2.toFixed(1)} lbs</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">Days Since Last Workout</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.timeSinceLastWorkout.toFixed(1)} days</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">Days Since Same Ex</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.timeSinceLastSameExercise.toFixed(1)} days</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">Body Weight</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.bodyWeight.toFixed(1)} lbs</span>
                    </div>
                    <div className="derived-item">
                      <span className="derived-label">BMI</span>
                      <span className="derived-value">{predictionResult.derivedFeatures.bmi.toFixed(1)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
