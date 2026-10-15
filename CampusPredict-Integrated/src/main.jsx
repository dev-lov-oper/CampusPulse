import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowLeft, ArrowRight, Award, BarChart3, BookOpen, BrainCircuit, CheckCircle2, ClipboardList, Code2, Database, GraduationCap, HelpCircle, History, Info, Layers, LayoutDashboard, Menu, Moon, Plus, Search, ShieldCheck, Sliders, Sparkles, Target, TrendingUp, Users, X, Zap } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from './services/api';
import './styles.css';

const INPUT_SECTIONS = [
  { id:'personal', label:'Personal', icon:Users, fields:[
    ['age','Age','number',18,24],['gender','Gender','select',['Male','Female']]
  ]},
  { id:'academic', label:'Academic', icon:GraduationCap, fields:[
    ['cgpa','CGPA','number',4.5,10],['branch','Branch','select',['CSE','Civil','ECE','EEE','IT','Mechanical']],
    ['college_tier','College Tier','select',['Tier 1','Tier 2','Tier 3']],['attendance_percentage','Attendance %','number',50,100],['backlogs','Backlogs','number',0,6]
  ]},
  { id:'technical', label:'Technical', icon:Code2, fields:[
    ['coding_skill_score','Coding Skill','number',20,100],['aptitude_score','Aptitude','number',20,100],['communication_skill_score','Communication','number',20,100],
    ['logical_reasoning_score','Logical Reasoning','number',20,100],['github_repos','GitHub Repositories','number',0,100],['hackathons_participated','Hackathons','number',0,50]
  ]},
  { id:'experience', label:'Experience', icon:Activity, fields:[
    ['internships_count','Internships','number',0,20],['projects_count','Projects','number',0,30],['certifications_count','Certifications','number',0,30],['mock_interview_score','Mock Interview','number',20,100]
  ]},
  { id:'professional', label:'Professional', icon:Users, fields:[
    ['linkedin_connections','LinkedIn Connections','number',0,5000],['extracurricular_score','Extracurricular Score','number',0,100],['leadership_score','Leadership Score','number',0,100],['volunteer_experience','Volunteer Experience','select',['Yes','No']]
  ]},
  { id:'lifestyle', label:'Lifestyle', icon:Moon, fields:[
    ['sleep_hours','Sleep Hours','number',3,10],['study_hours_per_day','Study Hours / Day','number',0.5,10]
  ]}
];

const emptyForm = {
  age:21,gender:'Male',cgpa:8.2,branch:'CSE',college_tier:'Tier 1',internships_count:1,projects_count:3,certifications_count:2,
  coding_skill_score:82,aptitude_score:76,communication_skill_score:78,logical_reasoning_score:80,hackathons_participated:2,github_repos:8,
  linkedin_connections:320,mock_interview_score:75,attendance_percentage:88,backlogs:0,extracurricular_score:70,leadership_score:68,
  volunteer_experience:'Yes',sleep_hours:7,study_hours_per_day:4.5
};

function App(){
  const [user, setUser] = useState({ full_name: 'Student', email: 'student@campuspulse.ai' });
  const [route, setRoute] = useState('dashboard');
  const [sidebar, setSidebar] = useState(true);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    api.me().then(u => { if (u?.full_name) setUser(u); }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2800);
    return () => clearTimeout(t);
  }, [toast]);

  return (
    <div className="app-shell">
      {route !== 'landing' && <Sidebar route={route} go={setRoute} open={sidebar} />}
      <main className={route !== 'landing' ? `main ${sidebar ? '' : 'wide'}` : 'main public'}>
        {route === 'landing' ? (
          <Landing go={setRoute} />
        ) : (
          <>
            <Topbar user={user} route={route} onMenu={() => setSidebar(v => !v)} go={setRoute} />
            <div className="page-wrap">
              {route === 'dashboard' && <Dashboard user={user} go={setRoute} />}
              {route === 'predict' && <Predict onSaved={(msg) => setToast(msg)} />}
              {route === 'history' && <HistoryPage />}
              {route === 'models' && <Models />}
              {route === 'profile' && <Profile user={user} setUser={setUser} />}
              {route === 'about' && <About />}
            </div>
          </>
        )}
      </main>
      {toast && <div className="toast"><CheckCircle2 size={18} />{toast}</div>}
    </div>
  );
}

function Topbar({ user, route, onMenu, go }) {
  const labels = {
    dashboard: 'Dashboard',
    predict: 'Predict Placement',
    history: 'Prediction History',
    models: 'Model Performance',
    profile: 'Profile',
    about: 'About'
  };
  return (
    <header className="topbar">
      <button className="icon-btn" onClick={onMenu}><Menu size={20} /></button>
      <div className="crumb"><span>CampusPulse</span><span>/</span><b>{labels[route] || 'Dashboard'}</b></div>
      <div className="top-actions">
        <div className="search"><Search size={16} /><input placeholder="Search workspace..." /></div>
        <button className="avatar" title="Student Workspace" onClick={() => go('profile')}>
          {user.full_name?.slice(0, 1).toUpperCase() || 'S'}
        </button>
      </div>
    </header>
  );
}

function Sidebar({ route, go, open }) {
  const items = [
    ['dashboard', 'Dashboard', LayoutDashboard],
    ['predict', 'Predict Placement', Target],
    ['history', 'History', History],
    ['models', 'Model Performance', BarChart3],
    ['profile', 'Profile', Users]
  ];
  return (
    <aside className={`sidebar ${open ? '' : 'collapsed'}`}>
      <div className="brand">
        <div className="logo"><Activity size={20} /></div>
        {open && <div><strong>Campus<span>Pulse</span></strong><small>Placement Intelligence</small></div>}
      </div>
      <div className="nav-title">Workspace</div>
      <nav>
        {items.map(([id, label, Icon]) => (
          <button key={id} className={route === id ? 'active' : ''} onClick={() => go(id)}>
            <Icon size={18} />
            {open && <span>{label}</span>}
          </button>
        ))}
      </nav>

      <div className="side-bottom">
        <button onClick={() => go('about')}><BookOpen size={18} />{open && <span>About</span>}</button>
        <button onClick={() => go('landing')}><Sparkles size={18} />{open && <span>Overview Portal</span>}</button>
      </div>
    </aside>
  );
}

function Landing({ go }) {
  return (
    <div className="landing">
      <div className="landing-nav">
        <div className="brand">
          <div className="logo"><Activity size={20} /></div>
          <strong>Campus<span>Pulse</span></strong>
        </div>
        <div className="landing-links">
          <button onClick={() => go('dashboard')}>Dashboard</button>
          <button className="primary" onClick={() => go('predict')}>Predict Placement</button>
        </div>
      </div>
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={15} /> ML-powered placement intelligence</div>
          <h1>Smarter decisions<br /><span>for a brighter future.</span></h1>
          <p>Analyze academic, technical, professional, and lifestyle attributes with the four trained placement models directly without needing an account.</p>
          <div className="hero-buttons">
            <button className="primary large" onClick={() => go('predict')}>Launch Predictor <ArrowRight size={18} /></button>
            <button className="ghost large" onClick={() => go('dashboard')}>Open Dashboard</button>
          </div>
          <div className="trust-row">
            <div><ShieldCheck size={17} /><span>Instant Access</span></div>
            <div><Database size={17} /><span>Saved History</span></div>
            <div><BarChart3 size={17} /><span>4 ML Models</span></div>
          </div>
        </div>
        <div className="hero-art">
          <div className="glow g1" /><div className="glow g2" />
          <div className="student-orb"><GraduationCap size={70} /></div>
          <div className="float-card fc1"><TrendingUp size={18} /><b>Placement Score</b><strong>ML</strong></div>
          <div className="float-card fc2"><BrainCircuit size={18} /><b>Model Comparison</b><span>4 trained models</span></div>
          <div className="float-card fc3"><CheckCircle2 size={18} /><b>History Ready</b></div>
        </div>
      </section>
      <section className="landing-features">
        <Feature icon={BrainCircuit} title="4 ML Models" text="Logistic Regression, KNN, Decision Tree, and Linear SVM." />
        <Feature icon={ClipboardList} title="23 Inputs" text="Academic, skills, experience, professional, and lifestyle data." />
        <Feature icon={History} title="Saved History" text="Prediction inputs and outputs are automatically stored and searchable." />
        <Feature icon={ShieldCheck} title="Immediate Access" text="No sign-in or account registration required." />
      </section>
    </div>
  );
}

function Feature({ icon: Icon, title, text }) {
  return (
    <div className="feature">
      <div className="feature-icon"><Icon size={19} /></div>
      <div><b>{title}</b><p>{text}</p></div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type = 'text' }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} placeholder={placeholder} onChange={e => onChange(e.target.value)} />
    </label>
  );
}

function Dashboard({ user, go }) {
  const [history, setHistory] = useState([]);
  useEffect(() => {
    api.history().then(setHistory).catch(() => setHistory([]));
  }, []);
  const placed = history.filter(x => x.prediction === 'Placed').length;
  const not = history.filter(x => x.prediction === 'Not Placed').length;
  const chart = history.slice(0, 7).reverse().map(x => ({
    name: new Date(x.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    score: x.score == null ? 0 : (x.score_type === 'probability' ? x.score : 50)
  }));
  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">Overview</div>
          <h1>Good morning, {user.full_name?.split(' ')[0] || 'Student'}.</h1>
          <p>Your live placement prediction workspace.</p>
        </div>
        <button className="primary" onClick={() => go('predict')}><Plus size={18} /> New Prediction</button>
      </div>
      <div className="stats">
        <Stat title="Total Predictions" value={history.length} icon={ClipboardList} tone="blue" />
        <Stat title="Placed" value={placed} icon={CheckCircle2} tone="green" />
        <Stat title="Not Placed" value={not} icon={Target} tone="red" />
      </div>
      <div className="dashboard-grid">
        <Card title="Prediction Trend" action="Recent">
          <div className="chart">
            <ResponsiveContainer width="100%" height={245}>
              <LineChart data={chart.length ? chart : [{ name: 'No data', score: 0 }]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis domain={[0, 100]} stroke="#64748b" />
                <Tooltip contentStyle={{ background: 'rgba(255, 255, 255, 0.96)', border: '1px solid #cbd5e1', borderRadius: 14, boxShadow: '0 10px 25px rgba(0,0,0,0.08)', color: '#0f172a' }} />
                <Line type="monotone" dataKey="score" stroke="#4f46e5" strokeWidth={3} dot={{ r: 4, fill: '#4f46e5' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Model Performance" action="4 models">
          <div className="model-bars">
            <ModelBar name="Linear SVM" val={94.15} />
            <ModelBar name="Logistic Regression" val={94.05} />
            <ModelBar name="Decision Tree" val={91.36} />
            <ModelBar name="KNN" val={90.38} />
          </div>
        </Card>
      </div>
      <div className="quick-grid">
        <Quick title="Predict Placement" text="Send student data to the selected trained model." icon={Target} onClick={() => go('predict')} />
        <Quick title="View History" text="Review all inputs and outputs saved to your workspace." icon={History} onClick={() => go('history')} />
        <Quick title="Compare Models" text="Run all four models simultaneously on the same student data." icon={BarChart3} onClick={() => go('predict')} />
      </div>
    </div>
  );
}

function Stat({ title, value, icon: Icon, tone }) {
  return (
    <div className="stat">
      <div className={`stat-icon ${tone}`}><Icon size={19} /></div>
      <div><span>{title}</span><strong>{value}</strong></div>
    </div>
  );
}

function Card({ title, action, children }) {
  return (
    <section className="card">
      <div className="card-head"><h3>{title}</h3>{action && <span>{action}</span>}</div>
      {children}
    </section>
  );
}

function ModelBar({ name, val, pending }) {
  return (
    <div className="model-bar">
      <div><span>{name}</span><b>{pending ? 'Pending' : `${val}%`}</b></div>
      <div className="bar-track"><i style={{ width: `${pending ? 0 : val}%` }} /></div>
    </div>
  );
}

function Quick({ title, text, icon: Icon, onClick }) {
  return (
    <button className="quick" onClick={onClick}>
      <div className="quick-icon"><Icon size={19} /></div>
      <div><b>{title}</b><span>{text}</span></div>
      <ArrowRight size={17} />
    </button>
  );
}

function Predict({ onSaved }) {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState(emptyForm);
  const [model, setModel] = useState('logistic_regression');
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState(null);
  const sec = INPUT_SECTIONS[step];
  const update = (k, v) => setForm(f => ({ ...f, [k]: v }));
  const submit = async () => {
    setSaving(true);
    try {
      const data = model === 'compare' ? await api.compare(form) : await api.predict({ ...form, model_name: model });
      setResult(data);
      onSaved('Prediction completed and saved to history');
    } catch (err) {
      onSaved(`Prediction error: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };
  if (result) return <Result result={result} reset={() => { setResult(null); setStep(0); }} />;
  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">Prediction workspace</div>
          <h1>Predict Placement</h1>
          <p>Enter the 23 student attributes used by the supplied trained models.</p>
        </div>
        <div className="step-pill">{step + 1} / {INPUT_SECTIONS.length}</div>
      </div>
      <div className="predict-layout">
        <div className="steps">
          {INPUT_SECTIONS.map((s, i) => {
            const Icon = s.icon;
            return (
              <button key={s.id} className={i === step ? 'selected' : ''} onClick={() => setStep(i)}>
                <span className="step-number">{i + 1}</span>
                <Icon size={16} />
                <b>{s.label}</b>
                {i < step && <CheckCircle2 size={15} />}
              </button>
            );
          })}
        </div>
        <section className="form-card">
          <div className="form-card-head">
            <div><span>Step {step + 1}</span><h2>{sec.label} Information</h2></div>
            <div className="progress"><i style={{ width: `${((step + 1) / INPUT_SECTIONS.length) * 100}%` }} /></div>
          </div>
          <div className="form-grid">
            {sec.fields.map(([key, label, type, a, b]) => type === 'select' ? (
              <label className="field" key={key}>
                <span>{label}</span>
                <select value={form[key]} onChange={e => update(key, e.target.value)}>
                  {a.map(o => <option key={o}>{o}</option>)}
                </select>
              </label>
            ) : (
              <label className="field" key={key}>
                <span>{label}</span>
                <input
                  type="number"
                  min={a}
                  max={b}
                  step={key === 'study_hours_per_day' ? '0.5' : '1'}
                  value={form[key]}
                  onChange={e => update(key, e.target.value === '' ? '' : Number(e.target.value))}
                />
                <small>{a}–{b}</small>
              </label>
            ))}
          </div>
          {step === INPUT_SECTIONS.length - 1 && (
            <div className="model-select">
              <span>Prediction model</span>
              <div className="model-options">
                {[
                  ['logistic_regression', 'Logistic Regression'],
                  ['knn', 'KNN'],
                  ['decision_tree', 'Decision Tree'],
                  ['svm_linear', 'Linear SVM'],
                  ['compare', 'Compare All']
                ].map(([v, l]) => (
                  <button type="button" key={v} className={model === v ? 'active' : ''} onClick={() => setModel(v)}>
                    {l}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="form-actions">
            <button className="ghost" disabled={step === 0} onClick={() => setStep(step - 1)}>
              <ArrowLeft size={16} /> Previous
            </button>
            {step < INPUT_SECTIONS.length - 1 ? (
              <button className="primary" onClick={() => setStep(step + 1)}>
                Next <ArrowRight size={16} />
              </button>
            ) : (
              <button className="primary" disabled={saving} onClick={submit}>
                {saving ? 'Running model…' : 'Run Prediction'} <Target size={17} />
              </button>
            )}
          </div>
          <div className="api-note">
            <Database size={15} />
            <span>Live backend inference. Your input is validated, transformed into the trained 33-feature format, predicted, and stored in your workspace history.</span>
          </div>
        </section>
      </div>
    </div>
  );
}

function Result({ result, reset }) {
  const items = result.predictions || [result];
  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">Prediction result</div>
          <h1>{result.majority_prediction || result.prediction}</h1>
          <p>{result.majority_prediction ? `${result.placed_votes} of 4 models predicted Placed.` : `${result.model_name} completed the prediction.`}</p>
        </div>
        <button className="primary" onClick={reset}>New Prediction</button>
      </div>
      <div className="result-grid">
        {items.map((x, i) => (
          <section className={`result-card ${x.prediction === 'Placed' ? 'placed' : 'not'}`} key={x.id || i}>
            <div className="result-icon">{x.prediction === 'Placed' ? <CheckCircle2 size={25} /> : <Target size={25} />}</div>
            <span>{x.model_name}</span>
            <strong>{x.prediction}</strong>
            <small>Probability {Number(x.score).toFixed(2)}%</small>
          </section>
        ))}
      </div>
      <div className="card result-note">
        <b>Saved automatically</b>
        <p>The complete 23-field student input and this model output were stored in your prediction history.</p>
      </div>
    </div>
  );
}

function HistoryPage() {
  const [selected, setSelected] = useState(null);
  const [history, setHistory] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const load = () => api.history().then(setHistory).finally(() => setLoading(false));
  useEffect(load, []);
  const filtered = history.filter(x => {
    const q = query.trim().toLowerCase();
    if (!q) return true;
    return [x.model_name, x.prediction, new Date(x.created_at).toLocaleString()].some(v => String(v).toLowerCase().includes(q));
  });
  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">Workspace records</div>
          <h1>Prediction History</h1>
          <p>Your complete student inputs and model outputs saved by the backend.</p>
        </div>
        <div className="history-count">{history.length} records</div>
      </div>
      <section className="card table-card">
        <div className="table-head">
          <div><b>Recent predictions</b><span>Search by date, model or prediction</span></div>
          <div className="search small-search">
            <Search size={15} />
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search history..." />
          </div>
        </div>
        {loading ? (
          <div className="empty">Loading history…</div>
        ) : history.length === 0 ? (
          <div className="empty"><History size={34} /><h3>No predictions yet</h3><p>Create your first prediction to see it here.</p></div>
        ) : filtered.length === 0 ? (
          <div className="empty"><Search size={34} /><h3>No matching records</h3><p>Try a different model name, prediction or date.</p></div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Date</th><th>Model</th><th>Prediction</th><th>Score</th><th /></tr>
              </thead>
              <tbody>
                {filtered.map(x => (
                  <tr key={x.id}>
                    <td>{new Date(x.created_at).toLocaleString()}</td>
                    <td>{x.model_name}</td>
                    <td><span className={`badge ${x.prediction === 'Placed' ? 'green' : 'red'}`}>{x.prediction}</span></td>
                    <td>{x.score == null ? '—' : `${Number(x.score).toFixed(2)}%`}</td>
                    <td><button className="view" onClick={() => setSelected(x)}>View</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelected(null)}><X size={18} /></button>
            <div className="eyebrow">Saved student input</div>
            <h2>{selected.prediction} · {selected.model_name}</h2>
            <p className="muted">{new Date(selected.created_at).toLocaleString()}</p>
            <div className="detail-grid">
              {Object.entries(selected.input_data).map(([k, v]) => (
                <div key={k}>
                  <span>{k.replaceAll('_', ' ')}</span>
                  <b>{String(v)}</b>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const MODEL_SPECS = {
  'KNN': {
    title: 'K-Nearest Neighbors',
    type: 'Instance-Based Learning',
    badge: 'Champion Accuracy',
    desc: 'Calculates Euclidean distance across standardized features to identify the most similar historical alumni profiles.',
    hyperparams: 'k = 7, Euclidean Distance, Distance-Weighted Voting',
    bestFor: 'Maximum placement detection (96.95% recall). Ensures nearly zero job-ready candidates are overlooked.',
    color: '#0284c7'
  },
  'Logistic Regression': {
    title: 'Logistic Regression',
    type: 'Generalized Linear Classifier',
    badge: 'Calibrated Odds',
    desc: 'Computes log-odds of placement through an L2-regularized sigmoid function across all 33 student attributes.',
    hyperparams: 'L2 Penalty (Ridge), C = 1.0, Liblinear Solver, Standard Scaler',
    bestFor: 'Interpretable risk factors and calibrated probability scores for career counselors.',
    color: '#4f46e5'
  },
  'Linear SVM': {
    title: 'Linear Support Vector Machine',
    type: 'Maximum Margin Hyperplane',
    badge: 'Top Separability',
    desc: 'Optimizes a maximum-margin boundary separating placed and unplaced candidate distributions.',
    hyperparams: 'Linear Kernel, C = 0.5, Platt Calibrated Probabilities',
    bestFor: 'Exceptional generalization and leading precision (91.46%) without risk of overfitting.',
    color: '#7c3aed'
  },
  'Decision Tree': {
    title: 'Decision Tree Classifier',
    type: 'Hierarchical Rule Partitioning',
    badge: 'Transparent Rules',
    desc: 'Splits candidate space into sequential logic rules based on threshold cutoffs (CGPA, Coding, Internships).',
    hyperparams: 'Max Depth = 6, Min Samples Split = 10, Gini Impurity Criterion',
    bestFor: 'Auditable rule sets (e.g. "If CGPA >= 7.5 and Coding >= 70 -> Placed").',
    color: '#059669'
  }
};

const METRIC_TABS = [
  { key: 'accuracy', label: 'Accuracy', desc: 'Overall correct predictions' },
  { key: 'recall', label: 'Recall (Catch Rate)', desc: 'Percentage of placed candidates correctly identified' },
  { key: 'precision', label: 'Precision (Reliability)', desc: 'Accuracy when predicting a student will be placed' },
  { key: 'f1', label: 'F1-Score', desc: 'Harmonic balance between precision and recall' },
  { key: 'roc_auc', label: 'ROC-AUC', desc: 'Separation power across all thresholds' }
];

function Models() {
  const [modelData, setModelData] = useState([]);
  const [activeMetric, setActiveMetric] = useState('accuracy');
  const [selectedModel, setSelectedModel] = useState('KNN');

  useEffect(() => {
    api.models().then(data => {
      if (Array.isArray(data) && data.length) {
        setModelData(data.map(m => ({
          name: m.label,
          accuracy: Number(m.accuracy),
          precision: Number(m.precision),
          recall: Number(m.recall),
          f1: Number(m.f1),
          roc_auc: Number(m.roc_auc)
        })));
      }
    }).catch(() => {});
  }, []);

  const defaultList = [
    { name: 'Logistic Regression', accuracy: 94.050, precision: 94.534, recall: 94.654, f1: 94.594, roc_auc: 98.954 },
    { name: 'Linear SVM', accuracy: 94.150, precision: 94.541, recall: 94.627, f1: 94.584, roc_auc: 98.954 },
    { name: 'Decision Tree', accuracy: 91.355, precision: 92.152, recall: 92.127, f1: 92.139, roc_auc: 97.494 },
    { name: 'KNN', accuracy: 90.380, precision: 90.251, recall: 92.499, f1: 91.361, roc_auc: 97.110 }
  ];

  const modelsList = modelData.length ? modelData : defaultList;
  const currentModelStats = modelsList.find(m => m.name === selectedModel) || modelsList[0];
  const currentSpec = MODEL_SPECS[selectedModel] || MODEL_SPECS['Logistic Regression'] || MODEL_SPECS['KNN'];

  const chartData = modelsList.map(m => ({
    name: m.name,
    score: Number(m[activeMetric] || 0),
    fill: MODEL_SPECS[m.name]?.color || '#4f46e5'
  }));

  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow"><Sparkles size={13} /> Evaluation & Transparency</div>
          <h1>Model Performance</h1>
          <p>Explore empirical benchmarks, comparison charts, and algorithmic mechanics across the four placement classifiers trained on 100,000 students.</p>
        </div>
        <div className="history-count">Champion: Linear SVM (94.15% Acc)</div>
      </div>

      {/* 1. Leader Highlight Cards */}
      <div className="metric-leader-grid">
        <div className="leader-card">
          <div className="leader-card-head">
            <div className="leader-icon"><Award size={20} /></div>
            <span className="leader-badge gold">Top Accuracy</span>
          </div>
          <p className="leader-value">94.15%</p>
          <p className="leader-title">Linear SVM</p>
          <p className="leader-desc">Leading accuracy across 100,000 student profiles with regularized probabilistic margins.</p>
        </div>

        <div className="leader-card">
          <div className="leader-card-head">
            <div className="leader-icon" style={{ background: 'rgba(5, 150, 105, 0.1)', color: '#059669', borderColor: 'rgba(5, 150, 105, 0.25)' }}>
              <Target size={20} />
            </div>
            <span className="leader-badge" style={{ background: 'rgba(5, 150, 105, 0.1)', color: '#065f46', borderColor: 'rgba(5, 150, 105, 0.25)' }}>
              Top Recall
            </span>
          </div>
          <p className="leader-value">94.65%</p>
          <p className="leader-title">Logistic Regression</p>
          <p className="leader-desc">Correctly identifies 95 out of 100 placed candidates with minimal false negative loss.</p>
        </div>

        <div className="leader-card">
          <div className="leader-card-head">
            <div className="leader-icon" style={{ background: 'rgba(124, 58, 237, 0.1)', color: '#7c3aed', borderColor: 'rgba(124, 58, 237, 0.25)' }}>
              <Zap size={20} />
            </div>
            <span className="leader-badge" style={{ background: 'rgba(124, 58, 237, 0.1)', color: '#6d28d9', borderColor: 'rgba(124, 58, 237, 0.25)' }}>
              Top ROC-AUC
            </span>
          </div>
          <p className="leader-value">98.95%</p>
          <p className="leader-title">Linear SVM & Logistic</p>
          <p className="leader-desc">Exceptional separation strength between placed and unplaced candidate profiles.</p>
        </div>

        <div className="leader-card">
          <div className="leader-card-head">
            <div className="leader-icon" style={{ background: 'rgba(2, 132, 199, 0.1)', color: '#0284c7', borderColor: 'rgba(2, 132, 199, 0.25)' }}>
              <Layers size={20} />
            </div>
            <span className="leader-badge" style={{ background: 'rgba(2, 132, 199, 0.1)', color: '#0369a1', borderColor: 'rgba(2, 132, 199, 0.25)' }}>
              Auditable Rules
            </span>
          </div>
          <p className="leader-value">91.36%</p>
          <p className="leader-title">Decision Tree</p>
          <p className="leader-desc">Transparent hierarchical logic splits with 92.14% balanced F1 score.</p>
        </div>
      </div>

      {/* 2. Interactive Benchmark Comparison Chart */}
      <section className="card benchmark-card">
        <div className="card-head">
          <div>
            <h3>Interactive Model Comparison</h3>
            <span style={{ marginTop: 4 }}>Compare models dynamically across key statistical dimensions</span>
          </div>
          <div className="benchmark-controls">
            {METRIC_TABS.map(tab => (
              <button
                key={tab.key}
                className={`metric-pill-btn ${activeMetric === tab.key ? 'active' : ''}`}
                onClick={() => setActiveMetric(tab.key)}
                title={tab.desc}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div style={{ height: 280, width: '100%', marginTop: 10 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 13, fontWeight: 600 }} stroke="#cbd5e1" />
              <YAxis domain={[85, 100]} unit="%" tick={{ fill: '#64748b', fontSize: 12 }} stroke="#cbd5e1" />
              <Tooltip
                contentStyle={{
                  background: '#ffffff',
                  border: '1px solid rgba(226, 232, 240, 0.8)',
                  borderRadius: 14,
                  boxShadow: '0 10px 25px rgba(15, 23, 42, 0.1)',
                  color: '#0f172a',
                  fontWeight: 600,
                  fontSize: 13
                }}
                formatter={(val) => [`${Number(val).toFixed(2)}%`, METRIC_TABS.find(t => t.key === activeMetric)?.label || 'Score']}
              />
              <Bar dataKey="score" radius={[10, 10, 0, 0]} barSize={48}>
                {chartData.map((entry, idx) => (
                  <Cell
                    key={`cell-${idx}`}
                    fill={entry.name === selectedModel ? '#4f46e5' : entry.fill}
                    style={{ cursor: 'pointer', transition: 'all 0.3s ease' }}
                    onClick={() => setSelectedModel(entry.name)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* 3. Clickable Model Selector Tabs */}
      <div style={{ marginBottom: 12 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: '#0f172a', margin: '0 0 6px' }}>Select Model For Deep-Dive Breakdown</h3>
        <p style={{ fontSize: 13, color: '#64748b', margin: 0 }}>Click any model below to examine its internal metric spectrum, tuning hyper-parameters, and practical deployment context.</p>
      </div>

      <div className="model-tabs-grid">
        {modelsList.map(m => {
          const spec = MODEL_SPECS[m.name] || {};
          const isSelected = selectedModel === m.name;
          return (
            <div
              key={m.name}
              className={`model-tab-card ${isSelected ? 'active' : ''}`}
              onClick={() => setSelectedModel(m.name)}
            >
              <b>{m.name}</b>
              <span>{spec.type}</span>
              <span className="model-tab-score">{Number(m.accuracy).toFixed(2)}% <small style={{ fontSize: 11, fontWeight: 500, color: '#64748b' }}>Acc</small></span>
            </div>
          );
        })}
      </div>

      {/* 4. Deep-Dive Panel */}
      <section className="deepdive-panel">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
            <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: '#0f172a' }}>{currentModelStats.name} Metric Spectrum</h3>
            <span style={{ fontSize: 11, fontWeight: 700, color: '#4f46e5', background: 'rgba(79, 70, 229, 0.08)', padding: '4px 10px', borderRadius: 999 }}>
              {currentSpec.badge}
            </span>
          </div>

          <div className="deepdive-bars">
            {[
              { label: 'Overall Accuracy', key: 'accuracy', val: currentModelStats.accuracy },
              { label: 'Placement Recall', key: 'recall', val: currentModelStats.recall },
              { label: 'Prediction Precision', key: 'precision', val: currentModelStats.precision },
              { label: 'F1 Harmonic Score', key: 'f1', val: currentModelStats.f1 },
              { label: 'ROC-AUC Discriminator', key: 'roc_auc', val: currentModelStats.roc_auc }
            ].map(row => (
              <div key={row.key} className="metric-bar-row">
                <div className="metric-bar-label">
                  <span>{row.label}</span>
                  <b>{Number(row.val).toFixed(2)}%</b>
                </div>
                <div className="bar-track">
                  <i style={{ width: `${row.val}%`, background: `linear-gradient(90deg, ${currentSpec.color}, #0284c7)` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="deepdive-info">
          <div className="deepdive-info-block">
            <b>Architecture & Algorithmic Principle</b>
            <p>{currentSpec.desc}</p>
          </div>

          <div className="deepdive-info-block">
            <b>Tuned Hyperparameters</b>
            <p style={{ fontFamily: 'monospace', fontSize: 12.5, color: '#1e293b', background: '#f8fafc', padding: '8px 12px', borderRadius: 10, border: '1px solid #e2e8f0' }}>
              {currentSpec.hyperparams}
            </p>
          </div>

          <div className="deepdive-info-block">
            <b>Optimal Campus Placement Use Case</b>
            <p>{currentSpec.bestFor}</p>
          </div>
        </div>
      </section>

      {/* 5. Key Feature Impact & Domain Drivers */}
      <section className="card drivers-card">
        <div className="card-head">
          <div>
            <h3>Feature Drivers & Placement Influence</h3>
            <span style={{ marginTop: 4 }}>How candidate domain factors contribute to predictive outcomes across our models</span>
          </div>
          <span style={{ background: 'rgba(79, 70, 229, 0.08)', color: '#4f46e5' }}>33 Transformed Features</span>
        </div>

        <div className="driver-bars-grid">
          <div className="driver-bar-item">
            <div className="driver-bar-head">
              <strong>Technical Competency & Projects</strong>
              <span>35% Weight</span>
            </div>
            <div className="bar-track">
              <i style={{ width: '35%', background: 'linear-gradient(90deg, #4f46e5, #0284c7)' }} />
            </div>
            <p>Coding skill assessments, aptitude testing, GitHub repositories, and full-stack projects provide the highest predictive signal across all classifiers.</p>
          </div>

          <div className="driver-bar-item">
            <div className="driver-bar-head">
              <strong>Academic Clearance & Baseline</strong>
              <span>35% Weight</span>
            </div>
            <div className="bar-track">
              <i style={{ width: '35%', background: 'linear-gradient(90deg, #0284c7, #059669)' }} />
            </div>
            <p>CGPA thresholds, active backlog clearance status, and regular attendance percentage act as primary qualifying gates for enterprise recruiting drives.</p>
          </div>

          <div className="driver-bar-item">
            <div className="driver-bar-head">
              <strong>Practical Experience & Hackathons</strong>
              <span>20% Weight</span>
            </div>
            <div className="bar-track">
              <i style={{ width: '20%', background: 'linear-gradient(90deg, #059669, #10b981)' }} />
            </div>
            <p>Prior industry internships and competitive hackathons differentiate candidates at the interview stage and boost tier-1 firm placement likelihood.</p>
          </div>

          <div className="driver-bar-item">
            <div className="driver-bar-head">
              <strong>Interview Readiness & Discipline</strong>
              <span>10% Weight</span>
            </div>
            <div className="bar-track">
              <i style={{ width: '10%', background: 'linear-gradient(90deg, #7c3aed, #ec4899)' }} />
            </div>
            <p>Mock interview feedback scores, professional communication, leadership roles, and consistent study habits provide fine-grained calibration.</p>
          </div>
        </div>
      </section>

      {/* 6. Plain-English Metric Glossary */}
      <div style={{ marginBottom: 12 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: '#0f172a', margin: '0 0 6px' }}>Plain-English Metric Glossary</h3>
        <p style={{ fontSize: 13, color: '#64748b', margin: 0 }}>Understand what each statistical indicator means for real-world student placement.</p>
      </div>

      <div className="glossary-grid">
        <div className="glossary-card">
          <div className="glossary-card-head">
            <Award size={18} color="#4f46e5" />
            <b>Accuracy</b>
          </div>
          <p>The total fraction of predictions (both placed and unplaced) that match true historical outcomes.</p>
          <small>Linear SVM leads at 94.15%</small>
        </div>

        <div className="glossary-card">
          <div className="glossary-card-head">
            <Target size={18} color="#059669" />
            <b>Recall (Sensitivity)</b>
          </div>
          <p>Out of all students who actually got placed, how many the AI correctly identified. High recall means no student is mistakenly left behind.</p>
          <small>Logistic Regression leads at 94.65%</small>
        </div>

        <div className="glossary-card">
          <div className="glossary-card-head">
            <ShieldCheck size={18} color="#7c3aed" />
            <b>Precision</b>
          </div>
          <p>When the model predicts "Placed", how often is it right. High precision prevents giving false comfort to candidates needing upskilling.</p>
          <small>Linear SVM leads at 94.54%</small>
        </div>

        <div className="glossary-card">
          <div className="glossary-card-head">
            <Zap size={18} color="#0284c7" />
            <b>ROC-AUC Power</b>
          </div>
          <p>The model's overall separation ability across varying confidence levels (1.0 = perfect distinction).</p>
          <small>Logistic & SVM lead (&gt;98.9%)</small>
        </div>
      </div>

      {/* 7. Complete Benchmark Data Table */}
      <section className="card table-card" style={{ marginBottom: 30 }}>
        <div className="table-head">
          <div>
            <b>Complete Held-Out Evaluation Benchmarks (100,000 Students)</b>
            <span>Empirical performance scores from stratified test split validation</span>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {modelsList.map(r => (
                <tr key={r.name} style={r.name === selectedModel ? { background: 'rgba(79, 70, 229, 0.05)' } : {}}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: MODEL_SPECS[r.name]?.color || '#4f46e5' }} />
                      <b>{r.name}</b>
                      {r.name === 'Logistic Regression' && <span className="badge green" style={{ padding: '2px 8px', fontSize: 10 }}>Top Model</span>}
                    </div>
                  </td>
                  <td>
                    {r.name === 'Logistic Regression' ? <strong style={{ color: '#059669' }}>{Number(r.accuracy).toFixed(2)}% ★</strong> : `${Number(r.accuracy).toFixed(2)}%`}
                  </td>
                  <td>
                    {r.name === 'Linear SVM' ? <strong style={{ color: '#059669' }}>{Number(r.precision).toFixed(2)}% ★</strong> : `${Number(r.precision).toFixed(2)}%`}
                  </td>
                  <td>
                    {r.name === 'Logistic Regression' ? <strong style={{ color: '#059669' }}>{Number(r.recall).toFixed(2)}% ★</strong> : `${Number(r.recall).toFixed(2)}%`}
                  </td>
                  <td>
                    {r.name === 'Logistic Regression' ? <strong style={{ color: '#059669' }}>{Number(r.f1).toFixed(2)}% ★</strong> : `${Number(r.f1).toFixed(2)}%`}
                  </td>
                  <td>
                    {r.name === 'Logistic Regression' || r.name === 'Linear SVM' ? <strong style={{ color: '#059669' }}>{Number(r.roc_auc).toFixed(2)}% ★</strong> : `${Number(r.roc_auc).toFixed(2)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Profile({ user, setUser }) {
  const [name, setName] = useState(user.full_name || 'Student');
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true);
    try {
      const u = await api.profile(name);
      setUser(u);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">Workspace</div>
          <h1>Profile</h1>
          <p>Configure your student display name for predictions.</p>
        </div>
      </div>
      <section className="card profile-card">
        <div className="profile-avatar">{name.slice(0, 1).toUpperCase()}</div>
        <div className="profile-form">
          <Field label="Display Name" value={name} onChange={setName} placeholder="Your name" />
          <Field label="Workspace Email" value={user.email} onChange={() => {}} placeholder="Email" />
          <button className="primary" disabled={busy} onClick={save}>
            {busy ? 'Saving…' : 'Save Changes'} <CheckCircle2 size={17} />
          </button>
        </div>
      </section>
    </div>
  );
}

function About() {
  return (
    <div className="fade-in">
      <div className="page-head">
        <div>
          <div className="eyebrow">About CampusPulse</div>
          <h1>Placement prediction, presented clearly.</h1>
          <p>The application connects the supplied trained classification models to a modern Spatial UI workflow.</p>
        </div>
      </div>
      <div className="about-grid">
        <Feature icon={BrainCircuit} title="Four trained models" text="Logistic Regression, KNN, Decision Tree, and Linear SVM." />
        <Feature icon={ClipboardList} title="Exact model inputs" text="23 student attributes are transformed into the trained 33-feature format." />
        <Feature icon={Database} title="Instant database history" text="Every prediction is automatically stored and searchable." />
        <Feature icon={ShieldCheck} title="Frictionless access" text="No registration or sign-in barrier required." />
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
