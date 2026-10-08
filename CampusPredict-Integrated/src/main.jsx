
import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowLeft, ArrowRight, BarChart3, BookOpen, BrainCircuit, CheckCircle2, ClipboardList, Code2, Database, GraduationCap, History, LayoutDashboard, LogIn, LogOut, Menu, Moon, Plus, Search, ShieldCheck, Sparkles, Target, TrendingUp, Users, X } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
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
  const [user,setUser]=useState(null);
  const [route,setRoute]=useState('landing');
  const [authMode,setAuthMode]=useState('login');
  const [sidebar,setSidebar]=useState(true);
  const [toast,setToast]=useState(null);
  const [loading,setLoading]=useState(true);

  useEffect(()=>{
    const token=localStorage.getItem('cp_token');
    if(!token){setLoading(false);return;}
    api.me().then(u=>{setUser(u);setRoute('dashboard')}).catch(()=>localStorage.removeItem('cp_token')).finally(()=>setLoading(false));
  },[]);
  useEffect(()=>{if(!toast)return;const t=setTimeout(()=>setToast(null),2800);return()=>clearTimeout(t)},[toast]);

  const logout=async()=>{localStorage.removeItem('cp_token');setUser(null);setRoute('landing');setToast('Signed out successfully')};
  const onAuth=(data)=>{localStorage.setItem('cp_token',data.access_token);setUser(data.user);setRoute('dashboard');setToast('Welcome to CampusPredict')};

  if(loading)return <div className="loading-screen"><div className="logo"><BrainCircuit size={24}/></div><span>Loading CampusPredict…</span></div>;
  if(!user && (route==='login'||route==='register'))return <Auth mode={authMode} setMode={setAuthMode} onAuth={onAuth} />;
  return <div className="app-shell">
    {user&&<Sidebar route={route} go={setRoute} open={sidebar} logout={logout}/>}
    <main className={user?`main ${sidebar?'':'wide'}`:'main public'}>
      {!user?<Landing go={(r)=>{setAuthMode(r);setRoute(r)}}/>:<><Topbar user={user} route={route} onMenu={()=>setSidebar(v=>!v)} go={setRoute}/><div className="page-wrap">
        {route==='dashboard'&&<Dashboard user={user} go={setRoute}/>}
        {route==='predict'&&<Predict onSaved={(msg)=>setToast(msg)}/>}
        {route==='history'&&<HistoryPage/>}
        {route==='models'&&<Models/>}
        {route==='profile'&&<Profile user={user} setUser={setUser}/>}
        {route==='about'&&<About/>}
      </div></>}
    </main>
    {toast&&<div className="toast"><CheckCircle2 size={18}/>{toast}</div>}
  </div>
}

function Topbar({user,route,onMenu,go}){const labels={dashboard:'Dashboard',predict:'Predict Placement',history:'Prediction History',models:'Model Performance',profile:'Profile',about:'About'};return <header className="topbar">
  <button className="icon-btn" onClick={onMenu}><Menu size={20}/></button><div className="crumb"><span>CampusPredict</span><span>/</span><b>{labels[route]||'Dashboard'}</b></div>
  <div className="top-actions"><div className="search"><Search size={16}/><input placeholder="Search..."/></div><button className="avatar" onClick={()=>go('profile')}>{user.full_name?.slice(0,1).toUpperCase()||'U'}</button></div>
</header>}

function Sidebar({route,go,open,logout}){const items=[['dashboard','Dashboard',LayoutDashboard],['predict','Predict Placement',Target],['history','History',History],['models','Model Performance',BarChart3],['profile','Profile',Users]];return <aside className={`sidebar ${open?'':'collapsed'}`}>
  <div className="brand"><div className="logo"><BrainCircuit size={20}/></div>{open&&<div><strong>Campus<span>Predict</span></strong><small>Placement Intelligence</small></div>}</div>
  <div className="nav-title">Workspace</div><nav>{items.map(([id,label,Icon])=><button key={id} className={route===id?'active':''} onClick={()=>go(id)}><Icon size={18}/>{open&&<span>{label}</span>}</button>)}</nav>
  {open&&<div className="side-card"><Sparkles size={18}/><b>AI Placement Insight</b><span>Run the four supplied trained models from one prediction.</span><button onClick={()=>go('predict')}>Try it <ArrowRight size={14}/></button></div>}
  <div className="side-bottom"><button onClick={()=>go('about')}><BookOpen size={18}/>{open&&<span>About</span>}</button><button onClick={logout}><LogOut size={18}/>{open&&<span>Logout</span>}</button></div>
</aside>}

function Landing({go}){return <div className="landing">
  <div className="landing-nav"><div className="brand"><div className="logo"><BrainCircuit size={20}/></div><strong>Campus<span>Predict</span></strong></div><div className="landing-links"><button onClick={()=>go('login')}>Login</button><button className="primary" onClick={()=>go('register')}>Get Started</button></div></div>
  <section className="hero"><div className="hero-copy"><div className="eyebrow"><Sparkles size={15}/> ML-powered placement intelligence</div><h1>Smarter decisions<br/><span>for a brighter future.</span></h1><p>Analyze academic, technical, professional and lifestyle attributes with the four trained placement models.</p><div className="hero-buttons"><button className="primary large" onClick={()=>go('register')}>Get Started <ArrowRight size={18}/></button><button className="ghost large" onClick={()=>go('login')}>Sign In</button></div><div className="trust-row"><div><ShieldCheck size={17}/><span>Secure accounts</span></div><div><Database size={17}/><span>Saved history</span></div><div><BarChart3 size={17}/><span>4 ML models</span></div></div></div>
  <div className="hero-art"><div className="glow g1"/><div className="glow g2"/><div className="student-orb"><GraduationCap size={70}/></div><div className="float-card fc1"><TrendingUp size={18}/><b>Placement Score</b><strong>ML</strong></div><div className="float-card fc2"><BrainCircuit size={18}/><b>Model Comparison</b><span>4 trained models</span></div><div className="float-card fc3"><CheckCircle2 size={18}/><b>History Ready</b></div></div></section>
  <section className="landing-features"><Feature icon={BrainCircuit} title="4 ML Models" text="Logistic Regression, KNN, Decision Tree and Linear SVM."/><Feature icon={ClipboardList} title="23 Inputs" text="Academic, skills, experience, professional and lifestyle data."/><Feature icon={History} title="Saved History" text="Prediction inputs and outputs are stored for the signed-in user."/><Feature icon={ShieldCheck} title="Protected Access" text="JWT authentication keeps prediction history user-specific."/></section>
</div>}

function Feature({icon:Icon,title,text}){return <div className="feature"><div className="feature-icon"><Icon size={19}/></div><div><b>{title}</b><p>{text}</p></div></div>}

function Auth({mode,setMode,onAuth}){
  const [form,setForm]=useState({full_name:'',email:'',password:'',confirm:''});const [error,setError]=useState('');const [busy,setBusy]=useState(false);
  const submit=async(e)=>{e.preventDefault();setError('');if(mode==='register'&&form.password!==form.confirm)return setError('Passwords do not match.');setBusy(true);try{const data=mode==='register'?await api.register({full_name:form.full_name,email:form.email,password:form.password}):await api.login({email:form.email,password:form.password});onAuth(data)}catch(err){setError(err.message)}finally{setBusy(false)}};
  return <div className="auth-page"><div className="auth-art"><div className="brand"><div className="logo"><BrainCircuit size={20}/></div><strong>Campus<span>Predict</span></strong></div><div className="auth-art-copy"><div className="eyebrow"><Sparkles size={15}/> Student Placement Intelligence</div><h1>{mode==='login'?'Welcome back.':'Start your journey.'}</h1><p>Create predictions, compare the four supplied models and keep your student prediction history in one secure account.</p></div><div className="auth-mini"><Activity size={18}/><span>Real backend authentication + saved prediction history</span></div></div>
  <div className="auth-panel"><button className="back" onClick={()=>window.location.reload()}><ArrowLeft size={16}/> Back</button><div className="auth-box"><div className="auth-heading"><div className="logo small"><LogIn size={18}/></div><div><h2>{mode==='login'?'Sign in':'Create account'}</h2><p>{mode==='login'?'Access your CampusPredict dashboard.':'Create your student workspace.'}</p></div></div>
  <form onSubmit={submit}>{mode==='register'&&<Field label="Full Name" value={form.full_name} onChange={v=>setForm({...form,full_name:v})} placeholder="Your full name"/>}<Field label="Email" type="email" value={form.email} onChange={v=>setForm({...form,email:v})} placeholder="student@example.com"/><Field label="Password" type="password" value={form.password} onChange={v=>setForm({...form,password:v})} placeholder="••••••••"/>{mode==='register'&&<Field label="Confirm Password" type="password" value={form.confirm} onChange={v=>setForm({...form,confirm:v})} placeholder="••••••••"/>}{error&&<div className="error">{error}</div>}<button className="primary full" disabled={busy} type="submit">{busy?'Please wait…':mode==='login'?'Sign In':'Create Account'} <ArrowRight size={17}/></button></form>
  <div className="switch">{mode==='login'?'New here?':'Already have an account?'} <button onClick={()=>setMode(mode==='login'?'register':'login')}>{mode==='login'?'Create account':'Sign in'}</button></div><div className="demo-note"><ShieldCheck size={15}/> Passwords are hashed by the FastAPI backend; they are not stored in the browser.</div></div></div></div>
}

function Field({label,value,onChange,placeholder,type='text'}){return <label className="field"><span>{label}</span><input type={type} value={value} placeholder={placeholder} onChange={e=>onChange(e.target.value)}/></label>}

function Dashboard({user,go}){const [history,setHistory]=useState([]);useEffect(()=>{api.history().then(setHistory).catch(()=>setHistory([]))},[]);
  const placed=history.filter(x=>x.prediction==='Placed').length,not=history.filter(x=>x.prediction==='Not Placed').length;
  const chart=history.slice(0,7).reverse().map(x=>({name:new Date(x.created_at).toLocaleDateString(undefined,{month:'short',day:'numeric'}),score:x.score==null?0:(x.score_type==='probability'?x.score:50)}));
  return <div className="fade-in"><div className="page-head"><div><div className="eyebrow">Overview</div><h1>Good morning, {user.full_name?.split(' ')[0]||'Student'}.</h1><p>Your live placement prediction workspace.</p></div><button className="primary" onClick={()=>go('predict')}><Plus size={18}/> New Prediction</button></div>
  <div className="stats"><Stat title="Total Predictions" value={history.length} icon={ClipboardList} tone="blue"/><Stat title="Placed" value={placed} icon={CheckCircle2} tone="green"/><Stat title="Not Placed" value={not} icon={Target} tone="red"/></div>
  <div className="dashboard-grid"><Card title="Prediction Trend" action="Recent"><div className="chart"><ResponsiveContainer width="100%" height={245}><LineChart data={chart.length?chart:[{name:'No data',score:0}]}><CartesianGrid strokeDasharray="3 3" stroke="#1b3450"/><XAxis dataKey="name" stroke="#6d829c"/><YAxis domain={[0,100]} stroke="#6d829c"/><Tooltip contentStyle={{background:'#0a192c',border:'1px solid #27425f',borderRadius:12}}/><Line type="monotone" dataKey="score" stroke="#29a8ff" strokeWidth={3} dot={{r:4}}/></LineChart></ResponsiveContainer></div></Card>
  <Card title="Model Performance" action="4 models"><div className="model-bars"><ModelBar name="Logistic Regression" val={90.67}/><ModelBar name="KNN" val={91.67}/><ModelBar name="Decision Tree" val={89.67}/><ModelBar name="Linear SVM" val={90.67}/></div></Card></div>
  <div className="quick-grid"><Quick title="Predict Placement" text="Send student data to the selected trained model." icon={Target} onClick={()=>go('predict')}/><Quick title="View History" text="Review all inputs and outputs saved to your account." icon={History} onClick={()=>go('history')}/><Quick title="Compare Models" text="Run all four models on the same student data." icon={BarChart3} onClick={()=>go('predict')}/></div></div>
}

function Stat({title,value,icon:Icon,tone}){return <div className="stat"><div className={`stat-icon ${tone}`}><Icon size={19}/></div><div><span>{title}</span><strong>{value}</strong></div></div>}
function Card({title,action,children}){return <section className="card"><div className="card-head"><h3>{title}</h3>{action&&<span>{action}</span>}</div>{children}</section>}
function ModelBar({name,val,pending}){return <div className="model-bar"><div><span>{name}</span><b>{pending?'Pending':`${val}%`}</b></div><div className="bar-track"><i style={{width:`${pending?0:val}%`}}/></div></div>}
function Quick({title,text,icon:Icon,onClick}){return <button className="quick" onClick={onClick}><div className="quick-icon"><Icon size={19}/></div><div><b>{title}</b><span>{text}</span></div><ArrowRight size={17}/></button>}

function Predict({onSaved}){
  const [step,setStep]=useState(0);const [form,setForm]=useState(emptyForm);const [model,setModel]=useState('logistic_regression');const [saving,setSaving]=useState(false);const [result,setResult]=useState(null);const sec=INPUT_SECTIONS[step];
  const update=(k,v)=>setForm(f=>({...f,[k]:v}));
  const submit=async()=>{setSaving(true);try{const data=model==='compare'?await api.compare(form):await api.predict({...form,model_name:model});setResult(data);onSaved('Prediction completed and saved to history');}catch(err){onSaved(`Prediction error: ${err.message}`)}finally{setSaving(false)}};
  if(result)return <Result result={result} reset={()=>{setResult(null);setStep(0)}}/>;
  return <div className="fade-in"><div className="page-head"><div><div className="eyebrow">Prediction workspace</div><h1>Predict Placement</h1><p>Enter the 23 student attributes used by the supplied trained models.</p></div><div className="step-pill">{step+1} / {INPUT_SECTIONS.length}</div></div>
  <div className="predict-layout"><div className="steps">{INPUT_SECTIONS.map((s,i)=>{const Icon=s.icon;return <button key={s.id} className={i===step?'selected':''} onClick={()=>setStep(i)}><span className="step-number">{i+1}</span><Icon size={16}/><b>{s.label}</b>{i<step&&<CheckCircle2 size={15}/>}</button>})}</div>
  <section className="form-card"><div className="form-card-head"><div><span>Step {step+1}</span><h2>{sec.label} Information</h2></div><div className="progress"><i style={{width:`${((step+1)/INPUT_SECTIONS.length)*100}%`}}/></div></div>
  <div className="form-grid">{sec.fields.map(([key,label,type,a,b])=>type==='select'?<label className="field" key={key}><span>{label}</span><select value={form[key]} onChange={e=>update(key,e.target.value)}>{a.map(o=><option key={o}>{o}</option>)}</select></label>:<label className="field" key={key}><span>{label}</span><input type="number" min={a} max={b} step={key==='study_hours_per_day'?'0.5':'1'} value={form[key]} onChange={e=>update(key,e.target.value===''?'':Number(e.target.value))}/><small>{a}–{b}</small></label>)}</div>
  {step===INPUT_SECTIONS.length-1&&<div className="model-select"><span>Prediction model</span><div className="model-options">{[['logistic_regression','Logistic Regression'],['knn','KNN'],['decision_tree','Decision Tree'],['svm_linear','Linear SVM'],['compare','Compare All']].map(([v,l])=><button type="button" key={v} className={model===v?'active':''} onClick={()=>setModel(v)}>{l}</button>)}</div></div>}
  <div className="form-actions"><button className="ghost" disabled={step===0} onClick={()=>setStep(step-1)}><ArrowLeft size={16}/> Previous</button>{step<INPUT_SECTIONS.length-1?<button className="primary" onClick={()=>setStep(step+1)}>Next <ArrowRight size={16}/></button>:<button className="primary" disabled={saving} onClick={submit}>{saving?'Running model…':'Run Prediction'} <Target size={17}/></button>}</div>
  <div className="api-note"><Database size={15}/><span>Live backend inference. Your input is validated, sent to FastAPI, processed into the trained 33-feature format, predicted, and stored in your history.</span></div></section></div></div>
}

function Result({result,reset}){const items=result.predictions||[result];return <div className="fade-in"><div className="page-head"><div><div className="eyebrow">Prediction result</div><h1>{result.majority_prediction||result.prediction}</h1><p>{result.majority_prediction?`${result.placed_votes} of 4 models predicted Placed.`:`${result.model_name} completed the prediction.`}</p></div><button className="primary" onClick={reset}>New Prediction</button></div>
  <div className="result-grid">{items.map((x,i)=><section className={`result-card ${x.prediction==='Placed'?'placed':'not'}`} key={x.id||i}><div className="result-icon">{x.prediction==='Placed'?<CheckCircle2 size={25}/>:<Target size={25}/>}</div><span>{x.model_name}</span><strong>{x.prediction}</strong><small>{x.score_type==='probability'?`Probability ${Number(x.score).toFixed(2)}%`:`Probability ${Number(x.score).toFixed(2)}%`}</small></section>)}</div>
  <div className="card result-note"><b>Saved automatically</b><p>The complete 23-field student input and this model output were stored in your prediction history.</p></div></div>}

function HistoryPage(){
  const [selected,setSelected]=useState(null);
  const [history,setHistory]=useState([]);
  const [query,setQuery]=useState('');
  const [loading,setLoading]=useState(true);
  const load=()=>api.history().then(setHistory).finally(()=>setLoading(false));
  useEffect(load,[]);
  const filtered=history.filter(x=>{
    const q=query.trim().toLowerCase();
    if(!q)return true;
    return [x.model_name,x.prediction,new Date(x.created_at).toLocaleString()].some(v=>String(v).toLowerCase().includes(q));
  });
  return <div className="fade-in"><div className="page-head"><div><div className="eyebrow">Private records</div><h1>Prediction History</h1><p>Your complete student inputs and model outputs saved by the backend.</p></div><div className="history-count">{history.length} records</div></div>
  <section className="card table-card"><div className="table-head"><div><b>Recent predictions</b><span>Search by date, model or prediction</span></div><div className="search small-search"><Search size={15}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search history..."/></div></div>
  {loading?<div className="empty">Loading history…</div>:history.length===0?<div className="empty"><History size={34}/><h3>No predictions yet</h3><p>Create your first prediction to see it here.</p></div>:filtered.length===0?<div className="empty"><Search size={34}/><h3>No matching records</h3><p>Try a different model name, prediction or date.</p></div>:<div className="table-wrap"><table><thead><tr><th>Date</th><th>Model</th><th>Prediction</th><th>Score</th><th/></tr></thead><tbody>{filtered.map(x=><tr key={x.id}><td>{new Date(x.created_at).toLocaleString()}</td><td>{x.model_name}</td><td><span className={`badge ${x.prediction==='Placed'?'green':'red'}`}>{x.prediction}</span></td><td>{x.score==null?'—':`${Number(x.score).toFixed(2)}%`}</td><td><button className="view" onClick={()=>setSelected(x)}>View</button></td></tr>)}</tbody></table></div>}</section>
  {selected&&<div className="modal-backdrop" onClick={()=>setSelected(null)}><div className="modal" onClick={e=>e.stopPropagation()}><button className="modal-close" onClick={()=>setSelected(null)}><X size={18}/></button><div className="eyebrow">Saved student input</div><h2>{selected.prediction} · {selected.model_name}</h2><p className="muted">{new Date(selected.created_at).toLocaleString()}</p><div className="detail-grid">{Object.entries(selected.input_data).map(([k,v])=><div key={k}><span>{k.replaceAll('_',' ')}</span><b>{String(v)}</b></div>)}</div></div></div>}</div>
}

function Models(){
  const [modelData, setModelData] = useState([]);
  useEffect(()=>{
    api.models().then(data=>{
      if (Array.isArray(data) && data.length) {
        setModelData(data.map(m=>[m.label, m.accuracy, m.precision, m.recall, m.f1, m.roc_auc]));
      }
    }).catch(()=>{});
  },[]);

  const defaultMetrics = [
    ['Logistic Regression',90.667,90.964,92.073,91.515,96.077],
    ['KNN',91.667,88.827,96.951,92.711,96.933],
    ['Decision Tree',89.667,89.349,92.073,90.691,93.728],
    ['Linear SVM',90.667,91.463,91.463,91.463,96.167]
  ];
  const metrics = modelData.length ? modelData : defaultMetrics;
  const best = metrics.reduce((a,b)=>b[1]>a[1]?b:a);
  return <div className="fade-in">
    <div className="page-head"><div><div className="eyebrow">Model intelligence</div><h1>Model Performance</h1><p>Final held-out test metrics from the supplied trained models.</p></div><div className="history-count">Best accuracy: {best[1].toFixed(2)}%</div></div>
    <section className="card table-card"><div className="table-wrap"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{metrics.map(r=><tr key={r[0]}><td><b>{r[0]}</b></td>{r.slice(1).map((v,i)=><td key={i}>{Number(v).toFixed(2)}%</td>)}</tr>)}</tbody></table></div></section>
    <div className="model-insight-grid">
      <section className="card insight-card"><div className="insight-icon"><BarChart3 size={20}/></div><div><b>Highest accuracy</b><strong>KNN (Robust Distance)</strong><span>91.67% accuracy and 96.95% recall with outlier-robust distance calculation.</span></div></section>
      <section className="card insight-card"><div className="insight-icon"><Target size={20}/></div><div><b>Pruned decision rules</b><strong>Decision Tree</strong><span>89.67% accuracy with cost-complexity pruning and non-linear synergy splits.</span></div></section>
      <section className="card insight-card"><div className="insight-icon"><Activity size={20}/></div><div><b>Calibrated margins</b><strong>Linear SVM & Logistic</strong><span>90.67% accuracy with Platt probability calibration and L2 regularization.</span></div></section>
    </div>
    <div className="explain-grid"><Feature icon={BrainCircuit} title="Logistic Regression" text="Estimates the relationship between student attributes and the binary placement outcome."/><Feature icon={Users} title="KNN" text="Predicts from similar examples in the training data."/><Feature icon={Target} title="Decision Tree" text="Predicts using learned decision rules."/><Feature icon={Activity} title="Linear SVM" text="Learns a separating decision boundary between the two classes."/></div>
  </div>
}

function Profile({user,setUser}){const [name,setName]=useState(user.full_name||'Student');const [busy,setBusy]=useState(false);const save=async()=>{setBusy(true);try{const u=await api.profile(name);setUser(u)}finally{setBusy(false)}};return <div className="fade-in"><div className="page-head"><div><div className="eyebrow">Account</div><h1>Profile</h1><p>Manage your CampusPredict account.</p></div></div><section className="card profile-card"><div className="profile-avatar">{name.slice(0,1).toUpperCase()}</div><div className="profile-form"><Field label="Full Name" value={name} onChange={setName} placeholder="Your name"/><Field label="Email" value={user.email} onChange={()=>{}} placeholder="Email"/><button className="primary" disabled={busy} onClick={save}>{busy?'Saving…':'Save Changes'} <CheckCircle2 size={17}/></button></div></section></div>}

function About(){return <div className="fade-in"><div className="page-head"><div><div className="eyebrow">About CampusPredict</div><h1>Placement prediction, presented clearly.</h1><p>The application connects the supplied trained classification models to a secure web workflow.</p></div></div><div className="about-grid"><Feature icon={BrainCircuit} title="Four trained models" text="Logistic Regression, KNN, Decision Tree and Linear SVM."/><Feature icon={ClipboardList} title="Exact model inputs" text="23 student attributes are transformed to the trained 28-feature format."/><Feature icon={Database} title="Backend history" text="Every prediction is stored against the authenticated user."/><Feature icon={ShieldCheck} title="Responsible use" text="Predictions are academic estimates, not guarantees of employment."/></div></div>}

createRoot(document.getElementById('root')).render(<App/>);
