/* ═══════════════════════════════════════════════════════════════
   HAVEN WORKSPACE — app.js
   Combines: Tree Roadmap + Arrow Mindmap + Inline Editing + Media
═══════════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────────
   SECTION 1: DATA — HAVEN Pipeline Modules
───────────────────────────────────────────────── */
const MODULES_DEFAULT = [
  {
    id:'ingest', label:'🎬 Module 1: INGEST', color:'#6b7280', badge:'current', badgeLabel:'Done',
    detail:{
      tag:'📥 Ingest', tagColor:'#6b7280',
      title:'Module 1 — Video Ingest',
      subtitle:'Đọc và chuẩn hoá frame đầu vào từ camera / file / RTSP stream',
      io:{ input:'RTSP stream\nVideo file (.mp4, .avi)\nWebcam (cv2.VideoCapture)', output:'frame: np.ndarray (H,W,3)\ntimestamp: float\nfps_in: float' },
      specs:[{label:'FPS Target',val:'25–30',sub:'input'},{label:'Frame skip',val:'Dynamic',sub:'khi lag'},{label:'Buffer',val:'5 frames',sub:'ring'},{label:'Status',val:'✅ Done',sub:''}],
      insights:[{type:'success',text:'✅ Module đã hoàn thành. Focus cho Detection và ADL.'},{type:'info',text:'Frame drop có chủ đích khi pipeline lag — giữ frame mới nhất.'}],
      structure:'class IngestModule:\n    def __init__(source, size=(640,640))\n    def read_frame() -> Tuple[bool, np.ndarray]\n    def get_metadata() -> Dict\n    def release()',
      media:[]
    },
    children:[{id:'ingest-io',label:'I/O Specification',badge:'done'},{id:'ingest-buf',label:'Frame Buffer Strategy',badge:'done'}]
  },
  {
    id:'detect', label:'🔍 Module 2: DETECTION', color:'#3b82f6', badge:'current', badgeLabel:'Baseline',
    detail:{
      tag:'🔍 Detection', tagColor:'#3b82f6',
      title:'Module 2 — Object Detection',
      subtitle:'YOLOv11n-pose — detect người + objects + keypoints trong 1 pass',
      io:{ input:'frame: np.ndarray (640,640,3)\nconf_thresh: float = 0.25', output:'List[Detection(\n  bbox: xyxy,\n  conf: float,\n  cls: int\n)]' },
      specs:[{label:'mAP@0.5',val:'39.5',sub:'COCO val'},{label:'FPS',val:'120',sub:'T4 GPU'},{label:'Params',val:'2.6M',sub:''},{label:'Size',val:'5.4MB',sub:''}],
      primaryMetric:'mAP@0.5 (COCO)',
      benchmark:{
        cols:['Model','Year','Params(M)','mAP@0.5','FPS T4','Size(MB)','Status'],
        rows:[
          {cells:['YOLOv11n-pose','2024','2.6','39.5','120','5.4','✅ Current'],current:true},
          {cells:['YOLOv10n','2024','2.3','38.5','133','4.8','🔬 Plan'],reco:true},
          {cells:['Gold-YOLO-N','2023','5.6','39.6','108','11.2','🔬 Plan'],reco:true},
          {cells:['YOLOv9c','2024','25.3','53.0','60','50.7','⚠️ GPU only']},
          {cells:['RT-DETRv2-S','2024','20.0','48.1','80','40.1','📖 Read only']},
          {cells:['YOLO-World-S','2024','11.2','37.4','80','22.5','🔬 Zero-shot']},
        ]
      },
      insights:[
        {type:'warn',text:'⚠️ mAP 39.5 thấp với small objects. Fine-tune trên custom HAVEN dataset.'},
        {type:'info',text:'🎯 Benchmark Gold-YOLO-N: cùng size YOLOv11n nhưng GD Mechanism giảm FP.'},
        {type:'info',text:'⚡ YOLOv10n NMS-free giảm latency ~5ms/frame — tốt cho real-time pipeline.'}
      ],
      papers:[
        {title:'YOLOv10: Real-Time End-to-End Object Detection',venue:'NeurIPS 2024',key:'NMS-free dual head',link:'https://arxiv.org/abs/2405.14458',badge:'plan'},
        {title:'Gold-YOLO: Efficient Object Detector via Gather-and-Distribute',venue:'NeurIPS 2023',key:'GD Mechanism giảm information loss trong FPN',link:'https://arxiv.org/abs/2309.11331',badge:'plan'},
        {title:'RT-DETRv2: Improved Baseline with Bag-of-Freebies',venue:'CVPR 2024',key:'Transformer NMS-free',link:'https://arxiv.org/abs/2407.17140',badge:'read'},
        {title:'YOLO-World: Real-Time Open-Vocabulary Object Detection',venue:'CVPR 2024',key:'CLIP text guidance, zero-shot',link:'https://arxiv.org/abs/2401.17270',badge:'read'},
      ],
      structure:'class DetectionModule:\n    def __init__(model_path, conf=0.25, iou=0.45)\n    def detect(frame) -> List[Detection]\n    def warmup(n_runs=3)',
      media:[]
    },
    children:[
      {id:'det-current',label:'YOLOv11n-pose (current)',badge:'current'},
      {id:'det-papers',label:'📚 Paper Survey (4 papers)',badge:'plan'},
      {id:'det-bench',label:'📊 Benchmark Table',badge:'wip'},
      {id:'det-improve',label:'🛠 Improvement Plan',badge:'plan'},
    ]
  },
  {
    id:'tracking', label:'🏃 Module 3: TRACKING', color:'#8b5cf6', badge:'current', badgeLabel:'Baseline',
    detail:{
      tag:'🏃 Tracking', tagColor:'#8b5cf6',
      title:'Module 3 — Multi-Object Tracking',
      subtitle:'ByteTrack — theo dõi liên tục người qua nhiều frames',
      io:{input:'List[Detection]\nframe: np.ndarray\nframe_id: int',output:'List[Track(\n  track_id: int,\n  bbox: xyxy,\n  age: int\n)]'},
      specs:[{label:'Overhead',val:'<2ms',sub:'per frame'},{label:'Buffer',val:'30f',sub:'lost track'},{label:'track_thresh',val:'0.5',sub:''},{label:'HOTA',val:'63.1',sub:'MOT17'}],
      primaryMetric:'HOTA (MOT17)',
      benchmark:{
        cols:['Tracker','Year','HOTA↑','IDF1↑','MOTA↑','IDs↓','FPS','Appearance','Status'],
        rows:[
          {cells:['ByteTrack','2022','63.1','77.3','80.3','422','28','❌','✅ Current'],current:true},
          {cells:['OC-SORT','2023','63.9','76.4','78.0','341','26','❌','🔬 Plan']},
          {cells:['BoT-SORT','2022','65.0','77.9','80.5','367','22','✅ opt','🔬 Priority'],reco:true},
          {cells:['Deep OC-SORT','2023','64.9','80.6','79.4','257','18','✅','🔬 Plan']},
          {cells:['StrongSORT','2022','63.5','81.9','78.3','245','15','✅','📖 Slow']},
        ]
      },
      insights:[
        {type:'warn',text:'⚠️ ByteTrack: IDs=422 — khi 2 người đi qua nhau, ID bị hoán đổi.'},
        {type:'info',text:'🎯 BoT-SORT: camera HAVEN cố định → tắt CMC → overhead ≈ 0. Thêm OSNet-x0.25.'},
        {type:'success',text:'✅ Indoor ít người (~2-4) → ByteTrack đang đủ dùng, benchmark BoT-SORT để confirm.'}
      ],
      papers:[
        {title:'ByteTrack: Multi-Object Tracking by Associating Every Detection Box',venue:'ECCV 2022',key:'ALL detections, không cần ReID',link:'https://arxiv.org/abs/2110.06864',badge:'current'},
        {title:'OC-SORT: Observation-Centric SORT',venue:'CVPR 2023',key:'Observation-centric Kalman update',link:'https://arxiv.org/abs/2203.14360',badge:'plan'},
        {title:'BoT-SORT: Robust Associations Multi-Pedestrian Tracking',venue:'arXiv 2022',key:'CMC + ReID feature integration',link:'https://arxiv.org/abs/2206.14651',badge:'plan'},
      ],
      structure:'class TrackingModule:\n    def __init__(track_thresh=0.5, match_thresh=0.8, buffer=30)\n    def update(dets, frame) -> List[Track]\n    def reset()',
      media:[]
    },
    children:[
      {id:'track-current',label:'ByteTrack (current)',badge:'current'},
      {id:'track-papers',label:'📚 Paper Survey (4)',badge:'plan'},
      {id:'track-bench',label:'📊 HOTA Benchmark',badge:'wip'},
      {id:'track-improve',label:'🛠 → BoT-SORT + OSNet',badge:'plan'},
    ]
  },
  {
    id:'pose', label:'🦴 Module 4: POSE', color:'#14b8a6', badge:'wip', badgeLabel:'Weak',
    detail:{
      tag:'🦴 Pose', tagColor:'#14b8a6',
      title:'Module 4 — Pose Estimation',
      subtitle:'17 keypoints COCO — input quan trọng cho ADL/Event module',
      io:{input:'frame (640,640,3)\nhoặc person_crops: List[array]',output:'keypoints: ndarray\n  shape: (N_persons,17,3)\n  [x, y, conf]'},
      specs:[{label:'Keypoints',val:'17',sub:'COCO'},{label:'Min conf',val:'0.3',sub:'ADL'},{label:'FPS',val:'≥15',sub:'combined'},{label:'Metric',val:'OKS/PCK',sub:''}],
      benchmark:{
        cols:['Model','Year','COCO AP↑','Speed(ms)','Params(M)','Status'],
        rows:[
          {cells:['YOLOv11n-pose','2024','50.0','8.3','2.9','✅ Current'],current:true},
          {cells:['RTMPose-S','2023','71.2','6.3','5.5','🔬 Priority'],reco:true},
          {cells:['RTMPose-M','2023','75.3','11.1','13.6','🔬 Plan']},
          {cells:['DWPose-S','2023','71.4','7.1','5.5','🔬 Plan']},
        ]
      },
      insights:[
        {type:'warn',text:'⚠️ CRITICAL: AP=50 rất thấp → keypoint sai → ADL sai → fall miss → safety issue!'},
        {type:'info',text:'🎯 RTMPose-S: AP=71 (+21!) với speed tương đương. Upgrade là highest-priority.'},
        {type:'success',text:'✅ RTMPose là 2-stage nhưng detector đã có sẵn → overhead nhỏ.'}
      ],
      papers:[
        {title:'RTMPose: Real-Time Multi-Person Pose Estimation',venue:'arXiv 2023',key:'SimCC head, ONNX-friendly, production-ready',link:'https://arxiv.org/abs/2303.07399',badge:'plan'},
        {title:'DWPose: Effective Whole-body Pose via Two-stage Distillation',venue:'ICCV 2023',key:'Knowledge distillation, whole-body',link:'https://arxiv.org/abs/2307.15880',badge:'plan'},
      ],
      structure:'# 17 COCO Keypoints\n0:nose 1:L_eye 2:R_eye 3:L_ear 4:R_ear\n5:L_shoulder 6:R_shoulder 7:L_elbow 8:R_elbow\n9:L_wrist 10:R_wrist 11:L_hip 12:R_hip\n13:L_knee 14:R_knee 15:L_ankle 16:R_ankle',
      media:[]
    },
    children:[
      {id:'pose-current',label:'YOLOv11n-pose AP=50 (weak)',badge:'wip'},
      {id:'pose-rtmpose',label:'→ RTMPose-S AP=71 (priority)',badge:'plan'},
      {id:'pose-bench',label:'📊 OKS / PCK Benchmark',badge:'plan'},
    ]
  },
  {
    id:'adl', label:'⚡ Module 5: ADL / EVENT ⭐', color:'#f97316', badge:'paper', badgeLabel:'⭐ Paper',
    detail:{
      tag:'⚡ ADL / Event', tagColor:'#f97316',
      title:'Module 5 — Activity & Fall Detection',
      subtitle:'🌟 Module tiềm năng nhất để ra paper — geometry → GCN/Transformer',
      io:{input:'keypoints_buffer: List[ndarray]\n  shape: (T, 17, 3)\n  T = 15–30 frames',output:'state: Enum{standing, sitting,\n  lying, walking}\nevent: Enum{fall, sit_down,\n  stand_up, None}'},
      specs:[{label:'Window T',val:'15–30f',sub:'0.5–1s'},{label:'Fall-Recall',val:'≥90%',sub:'safety!'},{label:'Latency',val:'<10ms',sub:''},{label:'Classes',val:'7',sub:''}],
      benchmark:{
        cols:['Method','Type','NTU-60↑','NTU-120↑','Fall-Recall↑','FPS','Status'],
        rows:[
          {cells:['Geometry rules','Rule','~78%','—','~65%','>1000','✅ Current'],current:true},
          {cells:['ST-GCN','GCN','81.5%','70.7%','~85%','45','🔬 First target'],reco:true},
          {cells:['MS-G3D','GCN','91.5%','86.9%','~89%','30','🔬 Plan']},
          {cells:['CTR-GCN','GCN','92.4%','88.9%','~90%','35','🔬 Plan']},
          {cells:['SkateFormer','Transformer','94.2%','90.3%','~92%','20','📖 Upper bound']},
        ]
      },
      insights:[
        {type:'paper',text:'⭐ PAPER: Improvement geometry→ST-GCN/CTR-GCN trên HAVEN indoor dataset → contribution cho WACV/Sensors/ICIP.'},
        {type:'warn',text:'🚨 SAFETY: Fall-Recall ≥ 90% bắt buộc! Recall > Precision — thà báo sai còn hơn bỏ sót.'},
        {type:'info',text:'📦 Public datasets: UR Fall Detection, NTU RGB+D 120, Le2i Fall Detection.'},
        {type:'info',text:'📝 Custom dataset cần ≥100 clips/class, 7 classes, ghi tại HAVEN environment.'}
      ],
      papers:[
        {title:'ST-GCN: Spatial Temporal Graph Convolutional Networks',venue:'CVPR 2018',key:'Skeleton graph, spatial+temporal convolution — classic baseline',link:'https://arxiv.org/abs/1801.07455',badge:'plan'},
        {title:'CTR-GCN: Channel-wise Topology Refinement Graph Convolution',venue:'ICCV 2021',key:'Channel-wise topology, SOTA GCN',link:'https://arxiv.org/abs/2107.12213',badge:'plan'},
        {title:'SkateFormer: Skeletal-Temporal Transformer',venue:'ECCV 2024',key:'Skate-MSA 4-type partition, SOTA 2024',link:'https://arxiv.org/abs/2403.09508',badge:'read'},
      ],
      structure:'# ADL Classes (7)\n0:standing 1:sitting 2:lying 3:walking\n4:fall ← CRITICAL\n5:sit_down 6:stand_up\n\n# ST-GCN Input: (N,C,T,V,M)\n# N=batch C=3 T=30 V=17kp M=1person',
      media:[]
    },
    children:[
      {id:'adl-current',label:'Rule-based Geometry (brittle)',badge:'wip'},
      {id:'adl-dataset',label:'📦 Dataset Collection',badge:'plan'},
      {id:'adl-stgcn',label:'🔬 ST-GCN (First ML baseline)',badge:'plan'},
      {id:'adl-ctrgcn',label:'🔬 CTR-GCN (Target)',badge:'plan'},
      {id:'adl-bench',label:'📊 Fall-Recall / F1 Benchmark',badge:'plan'},
      {id:'adl-paper',label:'📝 Paper → WACV/Sensors',badge:'paper'},
    ]
  },
  {
    id:'reid', label:'👤 Module 6: ReID', color:'#ec4899', badge:'wip', badgeLabel:'Weak',
    detail:{
      tag:'👤 ReID', tagColor:'#ec4899',
      title:'Module 6 — Person Re-Identification',
      subtitle:'Gán global_id nhất quán qua camera và thời gian',
      io:{input:'face_crop: (112,112,3)\nperson_crop: (128,256,3)\ntrack_id: int',output:'global_id: str\nsimilarity: float\nidentity: Optional[str]'},
      specs:[{label:'Face metric',val:'TAR@FAR',sub:'target >95%'},{label:'Body',val:'Rank-1',sub:'Market-1501'},{label:'Latency',val:'<5ms',sub:'per person'},{label:'Status',val:'⚠️ Weak',sub:'color hist'}],
      benchmark:{
        cols:['Model','Task','Rank-1↑','Params(M)','Speed(ms)','Status'],
        rows:[
          {cells:['Color Histogram','Person','~30%','0','<0.5','✅ Current (Weak)'],current:true},
          {cells:['OSNet-x0.25','Person','84.9%','0.6','2','🔬 Plan'],reco:true},
          {cells:['OSNet-x1.0','Person','94.8%','2.2','4','🔬 Plan']},
          {cells:['InsightFace (Buffalo_L)','Face','99.83% LFW','24','4','🔬 Priority'],reco:true},
          {cells:['BPBReID','Person','95.2%','8.1','8','🔬 Occlusion']},
        ]
      },
      insights:[
        {type:'warn',text:'⚠️ Color Histogram Rank-1 ~30% — fail khi ánh sáng thay đổi hoặc quần áo trùng màu.'},
        {type:'info',text:'🎯 InsightFace Buffalo_L: plug-and-play, 99.83% LFW — standard cho production face recognition.'},
        {type:'info',text:'💡 OSNet-x0.25: 0.6M params, 2ms — Rank-1 84.9%, gấp 3 lần current. Implement ngay.'}
      ],
      papers:[
        {title:'OSNet: Omni-Scale Feature Learning for Person Re-Identification',venue:'ICCV 2019',key:'Multi-scale omni-scale feature, lightweight',link:'https://arxiv.org/abs/1905.00953',badge:'plan'},
        {title:'InsightFace / ArcFace',venue:'CVPR 2019',key:'Angular margin loss, SOTA face recognition',link:'https://arxiv.org/abs/1801.07698',badge:'plan'},
      ],
      structure:'# Quick integration\nimport insightface\napp = insightface.app.FaceAnalysis(name="buffalo_l")\napp.prepare(ctx_id=0)\nfaces = app.get(img)\n# face.embedding: 512-dim',
      media:[]
    },
    children:[
      {id:'reid-face',label:'😀 Face: InsightFace (priority)',badge:'plan'},
      {id:'reid-body',label:'🧍 Body: OSNet-x0.25',badge:'plan'},
      {id:'reid-current',label:'Color Histogram (weak)',badge:'wip'},
    ]
  },
  {
    id:'pipeline', label:'🔗 End-to-End Pipeline', color:'#22c55e', badge:'plan', badgeLabel:'Optimize',
    detail:{
      tag:'🔗 Pipeline', tagColor:'#22c55e',
      title:'End-to-End Pipeline Optimization',
      subtitle:'FPS budget và latency cho toàn HAVEN pipeline',
      io:{input:'1 frame @ 1080p\nfrom RTSP camera',output:'JSON event:\n{track_id, state, event,\nglobal_id, timestamp}'},
      specs:[{label:'Target FPS',val:'≥10',sub:'real-time'},{label:'Latency',val:'<100ms',sub:'end-to-end'},{label:'GPU',val:'RTX 3060',sub:'12GB'},{label:'Edge',val:'Jetson Nano',sub:'≥5 FPS'}],
      benchmark:{
        cols:['Module','Current(ms)','Target(ms)','Optimization'],
        rows:[
          {cells:['Ingest+Resize','5','3','Async decode']},
          {cells:['Detection (YOLOv11n)','8','6','TensorRT FP16']},
          {cells:['Pose (YOLOv11n-pose)','8','8','→ RTMPose-S']},
          {cells:['Tracking (ByteTrack)','2','2','OK']},
          {cells:['ADL (Geometry → ST-GCN)','<1','22','Batch per 5f']},
          {cells:['ReID (OSNet+InsightFace)','0.5','4','Every 15f']},
          {cells:['Logging/API','2','2','Async queue']},
          {cells:['TOTAL','~26ms (38FPS)','~45ms (22FPS)','After upgrades']},
        ]
      },
      insights:[
        {type:'success',text:'✅ Current ~26ms → 38 FPS. Tốt hơn target 10 FPS.'},
        {type:'warn',text:'⚠️ Sau ST-GCN (22ms thêm) → ~48ms → 21 FPS. Cần per-5-frame inference.'},
        {type:'info',text:'💡 Strategy: Detection+Pose mỗi frame; ADL mỗi 5f; ReID mỗi 15f.'}
      ],
      structure:'# FPS Budget Strategy\nDetection: every 1 frame\nPose:      every 1 frame\nTracking:  every 1 frame\nADL:       every 5 frames (buffer 30f)\nReID:      every 15 frames',
      media:[]
    },
    children:[
      {id:'pipe-fps',label:'⚡ FPS Budget Analysis',badge:'plan'},
      {id:'pipe-trt',label:'🚀 TensorRT Deployment',badge:'plan'},
      {id:'pipe-jetson',label:'📟 Jetson Nano Optimization',badge:'plan'},
    ]
  },
];

/* ─────────────────────────────────────────────────
   SECTION 2: TREE STATE
───────────────────────────────────────────────── */
function loadModules(){
  return JSON.parse(JSON.stringify(MODULES_DEFAULT));
}
function saveModules(){ /* in-memory only */ }
let MODULES = loadModules();

const tState = { collapsed:new Set(), selected:null, filter:'all', search:'' };
MODULES.forEach((m,i)=>{ if(i>0) tState.collapsed.add(m.id); });

/* ─────────────────────────────────────────────────
   SECTION 3: VIEW SWITCHING
───────────────────────────────────────────────── */
let currentView = 'tree';
function switchView(v){
  currentView = v;
  document.getElementById('view-tree').classList.toggle('hidden', v!=='tree');
  document.getElementById('view-map').classList.toggle('hidden', v!=='map');
  document.getElementById('tab-tree').classList.toggle('active', v==='tree');
  document.getElementById('tab-map').classList.toggle('active', v==='map');
  document.getElementById('hdr-tree-btns').classList.toggle('hidden', v!=='tree');
  document.getElementById('hdr-map-btns').classList.toggle('hidden', v!=='map');
  document.getElementById('searchBox').style.display = v==='tree' ? '' : 'none';
  if(v==='map'){ mmInitIfNeeded(); }
}

/* ─────────────────────────────────────────────────
   SECTION 4: TREE RENDER
───────────────────────────────────────────────── */
function buildTree(){
  const tree = document.getElementById('tree');
  tree.innerHTML='';
  const q = tState.search.toLowerCase();

  MODULES.forEach(mod=>{
    const isCollapsed = tState.collapsed.has(mod.id);
    const hasChildren = mod.children && mod.children.length>0;
    const filterOk = tState.filter==='all' || mod.badge===tState.filter ||
                     (mod.children||[]).some(c=>c.badge===tState.filter);
    const searchOk = !q || mod.label.toLowerCase().includes(q) ||
                     (mod.children||[]).some(c=>c.label.toLowerCase().includes(q));
    if(!filterOk && !searchOk && q) return;

    const section = document.createElement('div');
    section.className = 'tree-section';

    const header = document.createElement('div');
    header.className = 'tree-group-header'+(isCollapsed?' collapsed':'');
    header.innerHTML = `
      <span class="mod-dot" style="background:${mod.color}"></span>
      <span class="tgh-label">${mod.label}</span>
      <span class="ti-badge badge-${mod.badge}">${mod.badgeLabel}</span>
      ${hasChildren?'<span class="toggle-arrow">▾</span>':''}
    `;
    // Inline edit on double-click of label
    header.querySelector('.tgh-label').addEventListener('dblclick', e=>{
      e.stopPropagation();
      startInlineEdit(e.target, newVal=>{
        mod.label = newVal; saveModules(); buildTree();
      });
    });
    // Add child btn
    const addBtn = document.createElement('span');
    addBtn.className='hdr-add'; addBtn.title='Thêm mục con'; addBtn.textContent='+';
    addBtn.onclick = e=>{ e.stopPropagation(); openModal(mod.id); };
    header.appendChild(addBtn);

    header.addEventListener('click', e=>{
      if(e.target.classList.contains('hdr-add')) return;
      if(hasChildren) tState.collapsed.has(mod.id)?tState.collapsed.delete(mod.id):tState.collapsed.add(mod.id);
      if(mod.detail) selectDetail(mod);
      buildTree();
    });
    section.appendChild(header);

    if(hasChildren){
      const wrap = document.createElement('div');
      wrap.className='tree-children'+(isCollapsed?' collapsed':'');
      wrap.style.maxHeight = isCollapsed?'0':(mod.children.length*40+10)+'px';

      mod.children.forEach(child=>{
        const childOk = !q || child.label.toLowerCase().includes(q);
        const childFilterOk = tState.filter==='all'||child.badge===tState.filter;
        if(!childOk&&!childFilterOk&&q) return;

        const item = document.createElement('div');
        item.className='tree-item'+(tState.selected===child.id?' selected':'');
        item.innerHTML=`
          <span class="ti-icon" style="color:${mod.color}">└</span>
          <span class="ti-label">${child.label}</span>
          <span class="ti-badge badge-${child.badge}">${badgeLabel(child.badge)}</span>
        `;
        // Inline edit
        item.querySelector('.ti-label').addEventListener('dblclick', e=>{
          e.stopPropagation();
          startInlineEdit(e.target, newVal=>{
            child.label = newVal; saveModules(); buildTree();
          });
        });
        item.addEventListener('click', e=>{
          e.stopPropagation();
          tState.selected=child.id;
          selectDetail(child.detail?child:mod, child);
          buildTree();
        });
        wrap.appendChild(item);
      });
      section.appendChild(wrap);
    }
    tree.appendChild(section);
  });
}

function badgeLabel(b){
  return {current:'Now',plan:'Plan',done:'Done',wip:'WIP',paper:'Paper',hot:'Hot'}[b]||b;
}

/* Inline edit helper */
function startInlineEdit(el, onSave){
  const orig = el.textContent;
  el.setAttribute('contenteditable','true');
  el.focus();
  const sel = window.getSelection(), range=document.createRange();
  range.selectNodeContents(el); sel.removeAllRanges(); sel.addRange(range);
  const finish = (save)=>{
    el.removeAttribute('contenteditable');
    if(save && el.textContent.trim()) onSave(el.textContent.trim());
    else el.textContent=orig;
  };
  el.onblur=()=>finish(true);
  el.onkeydown=e=>{
    if(e.key==='Enter'){e.preventDefault();finish(true);}
    if(e.key==='Escape'){e.preventDefault();finish(false);}
  };
}

/* ─────────────────────────────────────────────────
   SECTION 5: DETAIL PANEL
───────────────────────────────────────────────── */
let _editingModuleId = null;
let _editingChildId  = null;

function selectDetail(item, child){
  tState.selected = child ? child.id : item.id;
  const d = item.detail||{};
  const el = document.getElementById('detail-content');

  if(!item.detail){
    el.innerHTML=`<div class="d-empty"><div class="d-empty-icon">📋</div><div style="font-size:16px;font-weight:600">Chi tiết đang cập nhật</div><div style="color:var(--muted)">Double-click label để đổi tên · nhấn <b>✏️</b> để thêm nội dung.</div></div>`;
    return;
  }

  // Track which module we're editing
  _editingModuleId = null; _editingChildId = null;
  MODULES.forEach(m=>{
    if(m.id===item.id) _editingModuleId=m.id;
    if(m.detail===item.detail) _editingModuleId=m.id;
  });

  let html = `<div class="d-header">
    <button class="d-edit-btn" onclick="openEditModal()">✏️ Chỉnh sửa</button>
    <div class="d-module-tag" style="background:${d.tagColor}22;color:${d.tagColor}">${d.tag||''}</div>
    <div class="d-title">${d.title||''}</div>
    <div class="d-subtitle">${d.subtitle||''}</div>
  </div>`;

  if(d.io){
    html+=`<div class="d-section"><div class="d-section-title">Input / Output</div>
      <div class="io-row">
        <div class="io-box"><div class="io-box-label">📥 Input</div><div class="io-box-val">${esc(d.io.input).replace(/\n/g,'<br>')}</div></div>
        <div class="io-box"><div class="io-box-label">📤 Output</div><div class="io-box-val">${esc(d.io.output).replace(/\n/g,'<br>')}</div></div>
      </div></div>`;
  }
  if(d.specs){
    html+=`<div class="d-section"><div class="d-section-title">Specifications</div>
      <div class="info-grid">${d.specs.map(s=>`<div class="info-card"><div class="info-card-label">${s.label}</div><div class="info-card-val">${s.val}</div><div class="info-card-sub">${s.sub}</div></div>`).join('')}</div></div>`;
  }
  if(d.insights&&d.insights.length){
    html+=`<div class="d-section"><div class="d-section-title">Key Insights</div>`;
    d.insights.forEach(ins=>{
      html+=`<div class="insight ${ins.type==='warn'?'warn':ins.type==='success'?'success':ins.type==='paper'?'paper':''}">${esc(ins.text)}</div>`;
    });
    html+=`</div>`;
  }
  if(d.benchmark){
    html+=`<div class="d-section"><div class="d-section-title">📊 Benchmark${d.primaryMetric?' — '+d.primaryMetric:''}</div>
      <div class="table-wrap"><table class="bench-table">
        <thead><tr>${d.benchmark.cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
        <tbody>${d.benchmark.rows.map(r=>`<tr class="${r.current?'row-current':r.reco?'row-reco':''}">${r.cells.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div></div>`;
  }
  if(d.papers&&d.papers.length){
    html+=`<div class="d-section"><div class="d-section-title">📚 Papers (${d.papers.length})</div>`;
    d.papers.forEach(p=>{
      html+=`<div class="paper-card">
        <div class="paper-card-head"><div><div class="paper-card-title">${esc(p.title)}</div><div class="paper-card-venue">${p.venue}</div></div><span class="ti-badge badge-${p.badge}">${badgeLabel(p.badge)}</span></div>
        <div class="paper-card-key">${esc(p.key)}</div>
        ${p.link?`<a class="paper-card-link" href="${p.link}" target="_blank">→ arXiv / Paper</a>`:''}
      </div>`;
    });
    html+=`</div>`;
  }
  if(d.structure){
    html+=`<div class="d-section"><div class="d-section-title">Code / Reference</div>
      <div class="code-block">${esc(d.structure)}</div></div>`;
  }
  // Media
  const media = d.media||[];
  if(media.length){
    html+=`<div class="d-section"><div class="d-section-title">📎 Media (${media.length})</div><div class="d-media-row">`;
    media.forEach(m=>{
      if(m.type==='image') html+=`<img class="d-media-img" src="${m.src}" alt="${m.alt||''}"/>`;
      else if(m.type==='video-file') html+=`<video class="d-media-video" src="${m.src}" controls></video>`;
      else if(m.type==='video-yt') html+=`<iframe class="d-media-yt" src="${m.src}" allowfullscreen></iframe>`;
    });
    html+=`</div></div>`;
  }

  el.innerHTML=html;
}

function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

/* ─────────────────────────────────────────────────
   SECTION 6: EDIT DETAIL MODAL
───────────────────────────────────────────────── */
let _emDetail = null; // reference to detail object being edited

function openEditModal(){
  const mod = MODULES.find(m=>m.id===_editingModuleId);
  if(!mod||!mod.detail) return;
  _emDetail = mod.detail;
  document.getElementById('em-title').value = _emDetail.title||'';
  document.getElementById('em-subtitle').value = _emDetail.subtitle||'';
  document.getElementById('em-input').value = _emDetail.io?_emDetail.io.input:'';
  document.getElementById('em-output').value = _emDetail.io?_emDetail.io.output:'';
  document.getElementById('em-insights').value = (_emDetail.insights||[]).map(i=>i.text).join('\n');
  document.getElementById('em-structure').value = _emDetail.structure||'';
  renderEmMediaPreview();
  document.getElementById('em-img-url').value='';
  document.getElementById('em-vid-url').value='';
  document.getElementById('edit-modal-backdrop').classList.add('visible');
}
function closeEditModal(){ document.getElementById('edit-modal-backdrop').classList.remove('visible'); _emDetail=null; }
function saveEditModal(){
  if(!_emDetail) return;
  _emDetail.title = document.getElementById('em-title').value;
  _emDetail.subtitle = document.getElementById('em-subtitle').value;
  if(!_emDetail.io) _emDetail.io={input:'',output:''};
  _emDetail.io.input  = document.getElementById('em-input').value;
  _emDetail.io.output = document.getElementById('em-output').value;
  const lines = document.getElementById('em-insights').value.split('\n').filter(l=>l.trim());
  _emDetail.insights = lines.map(l=>{
    const t = l.startsWith('⚠️')||l.startsWith('⚠') ? 'warn'
             : l.startsWith('✅') ? 'success'
             : l.startsWith('⭐')||l.startsWith('📝') ? 'paper' : 'info';
    return {type:t,text:l};
  });
  _emDetail.structure = document.getElementById('em-structure').value;
  saveModules();
  closeEditModal();
  // Re-render detail
  const mod = MODULES.find(m=>m.id===_editingModuleId);
  if(mod&&mod.detail) selectDetail(mod);
}

/* Media in edit modal */
function renderEmMediaPreview(){
  const preview = document.getElementById('em-media-preview');
  const media = _emDetail&&_emDetail.media||[];
  if(!media.length){ preview.innerHTML=''; return; }
  preview.innerHTML = `<div class="d-section-title" style="margin-bottom:6px">Media đã thêm (${media.length})</div>
    <div class="d-media-row">${media.map((m,i)=>
      `<div style="position:relative">
        ${m.type==='image'?`<img class="d-media-img" src="${m.src}" style="max-height:100px"/>`
         :m.type==='video-file'?`<video class="d-media-video" src="${m.src}" style="max-height:100px" controls></video>`
         :`<span style="font-size:12px;color:var(--muted)">🎬 ${m.src.slice(0,40)}…</span>`}
        <button onclick="removeMedia(${i})" style="position:absolute;top:2px;right:2px;background:rgba(218,54,51,.8);color:#fff;border:none;border-radius:4px;padding:1px 5px;cursor:pointer;font-size:11px">✕</button>
      </div>`
    ).join('')}</div>`;
}
function removeMedia(i){ if(_emDetail&&_emDetail.media){ _emDetail.media.splice(i,1); saveModules(); renderEmMediaPreview(); } }

function addMediaToDetail(type){
  if(!_emDetail){ _emDetail = {}; }
  if(!_emDetail.media) _emDetail.media=[];
  if(type==='image'){
    const url = document.getElementById('em-img-url').value.trim();
    if(!url){ alert('Nhập URL ảnh hoặc upload file.'); return; }
    _emDetail.media.push({type:'image',src:url,alt:''});
    document.getElementById('em-img-url').value='';
  } else if(type==='video'){
    const url = document.getElementById('em-vid-url').value.trim();
    if(!url){ alert('Nhập URL video hoặc upload file.'); return; }
    const isYt = url.includes('youtube.com')||url.includes('youtu.be');
    const ytSrc = isYt ? url.replace('watch?v=','embed/').replace('youtu.be/','youtube.com/embed/') : url;
    _emDetail.media.push({type:isYt?'video-yt':'video-file',src:ytSrc});
    document.getElementById('em-vid-url').value='';
  }
  renderEmMediaPreview();
}

function handleDetailImageUpload(e){
  const file = e.target.files[0]; if(!file) return;
  const reader = new FileReader();
  reader.onload = ev=>{
    document.getElementById('em-img-url').value = ev.target.result;
  };
  reader.readAsDataURL(file);
  e.target.value='';
}
function handleDetailVideoUpload(e){
  const file = e.target.files[0]; if(!file) return;
  const url = URL.createObjectURL(file);
  document.getElementById('em-vid-url').value = url;
  e.target.value='';
}

/* ─────────────────────────────────────────────────
   SECTION 7: ADD CHILD MODAL
───────────────────────────────────────────────── */
let _modalModId = null;
function openModal(modId){
  _modalModId=modId;
  const mod=MODULES.find(m=>m.id===modId);
  document.getElementById('modal-heading').textContent=`Thêm mục con vào "${mod?.label||modId}"`;
  document.getElementById('modal-text').value='';
  document.getElementById('modal-badge').value='plan';
  document.getElementById('modal-backdrop').classList.add('visible');
  setTimeout(()=>document.getElementById('modal-text').focus(),50);
}
function closeModal(){ document.getElementById('modal-backdrop').classList.remove('visible'); _modalModId=null; }
function confirmModal(){
  const label=document.getElementById('modal-text').value.trim();
  if(!label) return;
  const badge=document.getElementById('modal-badge').value;
  const mod=MODULES.find(m=>m.id===_modalModId);
  if(!mod) return;
  if(!mod.children) mod.children=[];
  mod.children.push({id:_modalModId+'-'+Date.now(), label, badge, detail:null});
  saveModules(); closeModal();
  tState.collapsed.delete(_modalModId);
  buildTree();
}
document.getElementById('modal-text').addEventListener('keydown',e=>{
  if(e.key==='Enter') confirmModal();
  if(e.key==='Escape') closeModal();
});

/* ─────────────────────────────────────────────────
   SECTION 8: HEADER CONTROLS (tree view)
───────────────────────────────────────────────── */
document.getElementById('btnExpandAll').onclick=()=>{ tState.collapsed.clear(); buildTree(); };
document.getElementById('btnCollapseAll').onclick=()=>{ MODULES.forEach(m=>tState.collapsed.add(m.id)); buildTree(); };
document.getElementById('searchBox').addEventListener('input',e=>{
  tState.search=e.target.value.toLowerCase().trim();
  if(!tState.search) tState.collapsed=new Set(MODULES.slice(1).map(m=>m.id));
  else tState.collapsed.clear();
  buildTree();
});
function setFilter(btn,f){
  tState.filter=f;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  buildTree();
}
function toggleTheme(){
  const isDark=document.documentElement.getAttribute('data-theme')==='dark';
  document.documentElement.setAttribute('data-theme',isDark?'light':'dark');
  document.getElementById('btnTheme').textContent=isDark?'🌙 Dark':'☀ Light';
}

/* ─────────────────────────────────────────────────
   SECTION 9: MINDMAP ENGINE
───────────────────────────────────────────────── */
let mmInited = false;
let mmData = { nodes:{}, links:[], config:{curve:true} };
let mmView = { panX:0, panY:0, zoom:1 };
let mmSelectedNode = null;
let mmDragNode = null, mmDragStartX=0,mmDragStartY=0,mmDragOX=0,mmDragOY=0;
let mmConnDrag = null; // {sourceId, sourceSide, lastMouse}
let mmEpDrag   = null; // {edgeId, endpoint, lastMouse}
let mmBgDrag   = false, mmBgLastX=0,mmBgLastY=0;
let mmPendingDrag=null;

const mmWrap  = ()=>document.getElementById('mm-viewport-wrap');
const mmVP    = ()=>document.getElementById('mm-viewport');
const mmSvg   = ()=>document.getElementById('mm-svg');
const mmNodes = ()=>document.getElementById('mm-nodes');

function mmSaveState(){ /* in-memory only */ }
function mmLoadState(){ /* in-memory only */ }
function mmNormalize(){
  if(!mmData||typeof mmData!=='object') mmData={nodes:{},links:[],config:{curve:true}};
  if(!mmData.nodes) mmData.nodes={};
  if(!Array.isArray(mmData.links)) mmData.links=[];
  if(!mmData.config) mmData.config={curve:true};
  Object.values(mmData.nodes).forEach(n=>{
    if(typeof n.text!=='string') n.text=String(n.text||'');
    if(typeof n.collapsed!=='boolean') n.collapsed=false;
    if(!n.color) n.color='';
    if(!Array.isArray(n.media)) n.media=[];
  });
  mmData.links=mmData.links.filter(l=>l&&l.sourceId&&l.targetId&&mmData.nodes[l.sourceId]&&mmData.nodes[l.targetId]);
}

function mmInitIfNeeded(){
  if(mmInited) return;
  mmInited=true;
  mmLoadState();
  if(Object.keys(mmData.nodes).length===0) mmAddRootNode('HAVEN Pipeline');
  mmSetupEvents();
  mmRenderAll();
  mmFitView();
}

/* ── MM Apply Transform ── */
function mmApplyTransform(){
  mmVP().style.transform=`translate(${mmView.panX}px,${mmView.panY}px) scale(${mmView.zoom})`;
  document.getElementById('mm-container').style.setProperty('--mm-gx',mmView.panX+'px');
  document.getElementById('mm-container').style.setProperty('--mm-gy',mmView.panY+'px');
}
function mmClientToMap(cx,cy){
  const rect=mmWrap().getBoundingClientRect();
  return { x:(cx-rect.left-mmView.panX)/mmView.zoom, y:(cy-rect.top-mmView.panY)/mmView.zoom };
}

/* ── MM Events ── */
function mmSetupEvents(){
  const wrap=mmWrap();
  wrap.addEventListener('mousedown',e=>{
    if(e.target===wrap||e.target===mmVP()||e.target===mmSvg()||e.target.tagName==='svg'){
      mmBgDrag=true; mmBgLastX=e.clientX; mmBgLastY=e.clientY;
      wrap.classList.add('panning'); mmSelectNode(null);
    }
  });
  window.addEventListener('mousemove',e=>{
    if(mmConnDrag){
      mmConnDrag.lastMouse=mmClientToMap(e.clientX,e.clientY);
      mmRenderPaths(); return;
    }
    if(mmEpDrag){
      mmEpDrag.lastMouse=mmClientToMap(e.clientX,e.clientY);
      mmRenderPaths(); return;
    }
    if(mmPendingDrag&&!mmDragNode){
      const dx=e.clientX-mmPendingDrag.sx, dy=e.clientY-mmPendingDrag.sy;
      if(Math.hypot(dx,dy)>=4){
        mmDragNode=mmPendingDrag.id;
        mmDragStartX=mmPendingDrag.sx; mmDragStartY=mmPendingDrag.sy;
        mmDragOX=mmPendingDrag.ox; mmDragOY=mmPendingDrag.oy;
        mmPendingDrag=null;
      }
      return;
    }
    if(mmDragNode){
      const dx=(e.clientX-mmDragStartX)/mmView.zoom, dy=(e.clientY-mmDragStartY)/mmView.zoom;
      mmData.nodes[mmDragNode].x=mmDragOX+dx; mmData.nodes[mmDragNode].y=mmDragOY+dy;
      mmUpdateNodePos(mmDragNode); mmRenderPaths(); return;
    }
    if(mmBgDrag){
      mmView.panX+=e.clientX-mmBgLastX; mmView.panY+=e.clientY-mmBgLastY;
      mmBgLastX=e.clientX; mmBgLastY=e.clientY; mmApplyTransform();
    }
  });
  window.addEventListener('mouseup',e=>{
    if(mmConnDrag){
      const tEl=e.target.closest&&e.target.closest('.mm-node');
      const tId=tEl?tEl.dataset.id:null;
      if(tId&&tId!==mmConnDrag.sourceId){
        const sBox=mmGetNodeBox(mmConnDrag.sourceId), tBox=mmGetNodeBox(tId);
        if(sBox&&tBox){
          const sp=mmBoxPoint(sBox,mmConnDrag.sourceSide);
          const tSide=mmNearestSide(tBox,sp.x,sp.y);
          mmAddLink(mmConnDrag.sourceId,tId,mmConnDrag.sourceSide,tSide);
        }
      }
      mmConnDrag=null; mmRenderPaths();
    }
    if(mmEpDrag){
      const tEl=e.target.closest&&e.target.closest('.mm-node');
      const tId=tEl?tEl.dataset.id:null;
      const edge=mmData.links.find(l=>l.id===mmEpDrag.edgeId);
      if(edge&&tId){
        const fixedId=mmEpDrag.endpoint==='source'?edge.targetId:edge.sourceId;
        if(fixedId!==tId){
          const fixedBox=mmGetNodeBox(fixedId), tBox=mmGetNodeBox(tId);
          if(fixedBox&&tBox){
            const fp=mmEpDrag.endpoint==='source'?mmBoxPoint(fixedBox,edge.targetSide||'left'):mmBoxPoint(fixedBox,edge.sourceSide||'right');
            const ns=mmNearestSide(tBox,fp.x,fp.y);
            if(mmEpDrag.endpoint==='source'){ edge.sourceId=tId; edge.sourceSide=ns; }
            else { edge.targetId=tId; edge.targetSide=ns; }
            mmSaveState();
          }
        }
      }
      mmEpDrag=null; mmRenderPaths();
    }
    if(mmBgDrag){ mmBgDrag=false; mmWrap().classList.remove('panning'); }
    mmPendingDrag=null;
    if(mmDragNode){ mmDragNode=null; mmSaveState(); mmRenderAll(); }
  });
  wrap.addEventListener('wheel',e=>{
    e.preventDefault();
    const rect=wrap.getBoundingClientRect();
    const mx=e.clientX-rect.left, my=e.clientY-rect.top;
    const oldZ=mmView.zoom;
    let nz=oldZ*Math.exp(-e.deltaY*0.001);
    nz=Math.min(Math.max(nz,0.15),5);
    const ratio=nz/oldZ;
    mmView.panX=mx-(mx-mmView.panX)*ratio;
    mmView.panY=my-(my-mmView.panY)*ratio;
    mmView.zoom=nz; mmApplyTransform();
  });
  // Keyboard
  window.addEventListener('keydown',e=>{
    if(currentView!=='map') return;
    if(document.activeElement&&document.activeElement.isContentEditable) return;
    if(!mmSelectedNode) return;
    if(e.key==='Tab'||e.key==='Insert'){ e.preventDefault(); const c=mmAddChildNode(mmSelectedNode); if(c) mmEditNode(c); }
    if(e.key==='Enter'){ e.preventDefault(); const sib=mmAddSiblingNode(mmSelectedNode); if(sib) mmEditNode(sib); }
    if(e.key==='Delete'||e.key==='Backspace'){ if(!document.activeElement.isContentEditable){ e.preventDefault(); mmDeleteNode(mmSelectedNode); } }
    if(e.key===' '){ e.preventDefault(); mmToggleCollapse(mmSelectedNode); }
  });
}

/* ── MM Geometry Helpers ── */
function mmGetNodeBox(id){
  const el=document.querySelector(`.mm-node[data-id="${id}"]`); if(!el) return null;
  const n=mmData.nodes[id]; if(!n) return null;
  const w=el.offsetWidth, h=el.offsetHeight;
  return {x:n.x-w/2, y:n.y-h/2, w, h, cx:n.x, cy:n.y};
}
function mmBoxPoint(box,side){
  const cx=box.x+box.w/2, cy=box.y+box.h/2;
  return {
    top:{x:cx,y:box.y}, right:{x:box.x+box.w,y:cy},
    bottom:{x:cx,y:box.y+box.h}, left:{x:box.x,y:cy}
  }[side]||{x:cx,y:cy};
}
function mmNearestSide(box,px,py){
  const cx=box.x+box.w/2, cy=box.y+box.h/2;
  const pts={top:{x:cx,y:box.y},right:{x:box.x+box.w,y:cy},bottom:{x:cx,y:box.y+box.h},left:{x:box.x,y:cy}};
  let best='right',bd=Infinity;
  for(const[s,p] of Object.entries(pts)){
    const d=Math.hypot(px-p.x,py-p.y); if(d<bd){bd=d;best=s;}
  }
  return best;
}
function mmPathD(p1,p2,curve){
  if(!curve) return `M${p1.x},${p1.y} L${p2.x},${p2.y}`;
  const dx=(p2.x-p1.x)*0.5, dy=(p2.y-p1.y)*0.5;
  return `M${p1.x},${p1.y} C${p1.x+dx},${p1.y} ${p2.x-dx},${p2.y} ${p2.x},${p2.y}`;
}

/* ── MM Render ── */
function mmRenderAll(){ mmRenderNodes(); mmRenderPaths(); }
function mmUpdateNodePos(id){
  const el=document.querySelector(`.mm-node[data-id="${id}"]`); if(!el) return;
  const n=mmData.nodes[id]; el.style.left=n.x+'px'; el.style.top=n.y+'px';
}
function mmRenderNodes(){
  const layer=mmNodes(); layer.innerHTML='';
  Object.values(mmData.nodes).forEach(n=>{ if(!n.hidden) layer.appendChild(mmCreateNodeEl(n)); });
}
function mmCreateNodeEl(n){
  const el=document.createElement('div');
  el.className='mm-node'+(mmSelectedNode===n.id?' selected':'');
  el.dataset.id=n.id;
  el.style.cssText=`left:${n.x}px;top:${n.y}px;${n.color?'border-color:'+n.color:''}`;

  // Text
  const txt=document.createElement('div');
  txt.className='mm-node-text'; txt.textContent=n.text;
  el.appendChild(txt);

  // Media inside node
  if(n.media&&n.media.length){
    n.media.forEach(m=>{
      if(m.type==='image'){ const img=document.createElement('img'); img.src=m.src; img.draggable=false; el.appendChild(img); }
      else if(m.type==='video'){ const v=document.createElement('video'); v.src=m.src; v.controls=true; el.appendChild(v); }
    });
  }

  // Collapse indicator
  const childIds=Object.values(mmData.nodes).filter(c=>c.parentId===n.id).map(c=>c.id);
  if(childIds.length){
    const cBtn=document.createElement('div'); cBtn.className='mm-collapse'; cBtn.title='Thu/Mở';
    cBtn.textContent=n.collapsed?'▶':'◀';
    cBtn.onclick=e=>{e.stopPropagation();mmToggleCollapse(n.id);};
    el.appendChild(cBtn);
  }

  // Connection handles
  ['top','right','bottom','left'].forEach(side=>{
    const h=document.createElement('div');
    h.className='mm-handle'; h.dataset.s=side;
    h.addEventListener('mousedown',e=>{
      e.stopPropagation(); e.preventDefault();
      mmConnDrag={sourceId:n.id,sourceSide:side,lastMouse:mmClientToMap(e.clientX,e.clientY)};
    });
    el.appendChild(h);
  });

  // Drag
  el.addEventListener('mousedown',e=>{
    if(e.target.classList.contains('mm-handle')||e.target.classList.contains('mm-collapse')) return;
    e.stopPropagation();
    mmSelectNode(n.id);
    mmPendingDrag={id:n.id,sx:e.clientX,sy:e.clientY,ox:n.x,oy:n.y};
  });
  // Double-click to edit text
  txt.addEventListener('dblclick',e=>{
    e.stopPropagation(); mmEditNode(n.id);
  });
  // Right-click context menu
  el.addEventListener('contextmenu',e=>{
    e.preventDefault(); mmShowCtx(e,n.id);
  });

  return el;
}

function mmSelectNode(id){
  mmSelectedNode=id;
  document.querySelectorAll('.mm-node').forEach(el=>{
    el.classList.toggle('selected', el.dataset.id===id);
  });
}
function mmEditNode(id){
  const el=document.querySelector(`.mm-node[data-id="${id}"]`); if(!el) return;
  const txt=el.querySelector('.mm-node-text');
  mmSelectNode(id);
  el.classList.add('editing'); txt.contentEditable='true'; txt.focus();
  const sel=window.getSelection(),range=document.createRange();
  range.selectNodeContents(txt); sel.removeAllRanges(); sel.addRange(range);
  txt.onblur=()=>{
    el.classList.remove('editing'); txt.contentEditable='false';
    mmData.nodes[id].text=txt.textContent||''; mmSaveState();
  };
  txt.onkeydown=e=>{
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();txt.blur();}
    if(e.key==='Escape'){e.preventDefault();txt.textContent=mmData.nodes[id].text;txt.blur();}
  };
}

function mmRenderPaths(){
  const svg=mmSvg();
  // Remove all existing paths and ep handles
  svg.querySelectorAll('.mm-edge-path,.mm-ep,.mm-edge-label').forEach(el=>el.remove());
  const curve=mmData.config.curve;
  const marker='url(#mm-arrow)';
  const markerSel='url(#mm-arrow-sel)';

  // Tree edges (parent→child)
  const nodeIds=new Set(Object.keys(mmData.nodes));
  Object.values(mmData.nodes).forEach(n=>{
    if(!n.parentId||!nodeIds.has(n.parentId)) return;
    const pBox=mmGetNodeBox(n.parentId), cBox=mmGetNodeBox(n.id);
    if(!pBox||!cBox) return;
    const p1=mmBoxPoint(pBox,'right'), p2=mmBoxPoint(cBox,'left');
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.className.baseVal='mm-edge-path';
    path.setAttribute('d',mmPathD(p1,p2,curve));
    path.setAttribute('marker-end',marker);
    svg.appendChild(path);
  });

  // Custom links
  mmData.links.forEach(link=>{
    const sBox=mmGetNodeBox(link.sourceId), tBox=mmGetNodeBox(link.targetId);
    if(!sBox||!tBox) return;
    const p1=mmBoxPoint(sBox,link.sourceSide||'right'), p2=mmBoxPoint(tBox,link.targetSide||'left');
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.className.baseVal='mm-edge-path';
    path.setAttribute('d',mmPathD(p1,p2,curve));
    path.setAttribute('marker-end',marker);
    // Endpoint draggables
    [['source',p1,link.sourceId,link.sourceSide],['target',p2,link.targetId,link.targetSide]].forEach(([ep,pt,nid,side])=>{
      const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
      c.className.baseVal='mm-ep'; c.setAttribute('cx',pt.x); c.setAttribute('cy',pt.y); c.setAttribute('r','5');
      c.addEventListener('mousedown',e=>{
        e.stopPropagation(); e.preventDefault();
        mmEpDrag={edgeId:link.id,endpoint:ep,lastMouse:mmClientToMap(e.clientX,e.clientY)};
      });
      svg.appendChild(c);
    });
    svg.appendChild(path);
  });

  // Preview while dragging connection
  if(mmConnDrag&&mmConnDrag.lastMouse){
    const sBox=mmGetNodeBox(mmConnDrag.sourceId); if(!sBox) return;
    const p1=mmBoxPoint(sBox,mmConnDrag.sourceSide);
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.className.baseVal='mm-edge-path preview';
    path.setAttribute('d',mmPathD(p1,mmConnDrag.lastMouse,curve));
    svg.appendChild(path);
  }
  if(mmEpDrag&&mmEpDrag.lastMouse){
    const edge=mmData.links.find(l=>l.id===mmEpDrag.edgeId); if(!edge) return;
    const fixedId=mmEpDrag.endpoint==='source'?edge.targetId:edge.sourceId;
    const fixedSide=mmEpDrag.endpoint==='source'?edge.targetSide||'left':edge.sourceSide||'right';
    const fBox=mmGetNodeBox(fixedId); if(!fBox) return;
    const fp=mmBoxPoint(fBox,fixedSide);
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.className.baseVal='mm-edge-path preview';
    path.setAttribute('d',mmEpDrag.endpoint==='source'?mmPathD(mmEpDrag.lastMouse,fp,curve):mmPathD(fp,mmEpDrag.lastMouse,curve));
    svg.appendChild(path);
  }
}

/* ── MM Node Operations ── */
function mmGenId(){ return 'n'+Math.random().toString(36).substr(2,8); }
function mmGenLinkId(){ return 'l'+Math.random().toString(36).substr(2,8); }

function mmAddRootNode(text='Ý tưởng mới'){
  const id=mmGenId();
  const cx=(window.innerWidth/2-mmView.panX)/mmView.zoom;
  const cy=(window.innerHeight/2-mmView.panY)/mmView.zoom;
  mmData.nodes[id]={id,text,x:cx,y:cy,parentId:null,collapsed:false,media:[]};
  mmSelectNode(id); mmSaveState(); mmRenderAll(); mmEditNode(id);
}
function mmAddChildNode(parentId){
  const p=mmData.nodes[parentId]; if(!p) return null;
  p.collapsed=false;
  const id=mmGenId();
  const siblings=Object.values(mmData.nodes).filter(n=>n.parentId===parentId);
  mmData.nodes[id]={id,text:'Nhánh mới',x:p.x+220,y:p.y+siblings.length*70-siblings.length*35,parentId,collapsed:false,media:[]};
  mmAutoLayoutSiblings(parentId);
  mmSelectNode(id); mmSaveState(); mmRenderAll(); return id;
}
function mmAddSiblingNode(id){
  const n=mmData.nodes[id]; if(!n) return null;
  if(!n.parentId) return mmAddRootNode('Nhánh mới')||null;
  const p=mmData.nodes[n.parentId]; if(!p) return null;
  const sid=mmGenId();
  mmData.nodes[sid]={id:sid,text:'Nhánh mới',x:n.x,y:n.y+80,parentId:n.parentId,collapsed:false,media:[]};
  mmAutoLayoutSiblings(n.parentId);
  mmSelectNode(sid); mmSaveState(); mmRenderAll(); return sid;
}
function mmAutoLayoutSiblings(parentId){
  const p=mmData.nodes[parentId]; if(!p) return;
  const sibs=Object.values(mmData.nodes).filter(n=>n.parentId===parentId);
  const gap=80, startY=p.y-(sibs.length-1)*gap/2;
  sibs.forEach((s,i)=>{ s.y=startY+i*gap; s.x=p.x+220; });
}
function mmDeleteNode(id){
  if(!mmData.nodes[id]) return;
  // Recursively delete children
  const children=Object.values(mmData.nodes).filter(n=>n.parentId===id).map(n=>n.id);
  children.forEach(cid=>mmDeleteNode(cid));
  delete mmData.nodes[id];
  mmData.links=mmData.links.filter(l=>l.sourceId!==id&&l.targetId!==id);
  if(mmSelectedNode===id) mmSelectedNode=null;
  mmSaveState(); mmRenderAll();
}
function mmAddLink(sId,tId,sSide,tSide){
  const exists=mmData.links.some(l=>l.sourceId===sId&&l.targetId===tId);
  if(!exists) mmData.links.push({id:mmGenLinkId(),sourceId:sId,targetId:tId,sourceSide:sSide||'right',targetSide:tSide||'left'});
  mmSaveState(); mmRenderPaths();
}
function mmToggleCollapse(id){
  const n=mmData.nodes[id]; if(!n) return;
  n.collapsed=!n.collapsed;
  const hide=(nid,hidden)=>{
    const ch=Object.values(mmData.nodes).filter(c=>c.parentId===nid);
    ch.forEach(c=>{ c.hidden=hidden; hide(c.id,hidden); });
  };
  hide(id,n.collapsed);
  mmSaveState(); mmRenderAll();
}

/* ── MM FitView, Export ── */
function mmFitView(){
  const keys=Object.keys(mmData.nodes); if(!keys.length) return;
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  keys.forEach(id=>{ const n=mmData.nodes[id]; if(!n.hidden){ if(n.x<minX)minX=n.x; if(n.x>maxX)maxX=n.x; if(n.y<minY)minY=n.y; if(n.y>maxY)maxY=n.y; } });
  const W=mmWrap().offsetWidth||window.innerWidth, H=mmWrap().offsetHeight||window.innerHeight-56;
  const cw=maxX-minX+200, ch=maxY-minY+160;
  const zoom=Math.min(W/cw, H/ch, 1.2);
  mmView.zoom=zoom;
  mmView.panX=(W-(maxX+minX)*zoom)/2;
  mmView.panY=(H-(maxY+minY)*zoom)/2;
  mmApplyTransform();
}
function mmToggleLineType(){
  mmData.config.curve=!mmData.config.curve;
  document.getElementById('mm-btn-line').textContent=mmData.config.curve?'Cong':'Thẳng';
  mmSaveState(); mmRenderPaths();
}
function mmExportJSON(){
  const blob=new Blob([JSON.stringify(mmData,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='haven_mindmap.json'; a.click();
}
function mmImportJSON(e){
  const file=e.target.files[0]; if(!file) return;
  const reader=new FileReader();
  reader.onload=ev=>{ try{ mmData=JSON.parse(ev.target.result); mmNormalize(); mmSaveState(); mmRenderAll(); mmFitView(); }catch(err){ alert('File JSON không hợp lệ.'); } };
  reader.readAsText(file); e.target.value='';
}
function mmClearAll(){
  if(confirm('Xóa toàn bộ sơ đồ?')){ mmData={nodes:{},links:[],config:{curve:true}}; mmSaveState(); mmRenderAll(); }
}

/* ── MM Image / Video insert ── */
let mmPendingMediaNodeId=null;
function mmTriggerImageInsert(){
  if(!mmSelectedNode){ alert('Chọn một node trước, sau đó bấm 🖼 Ảnh.'); return; }
  mmPendingMediaNodeId=mmSelectedNode;
  document.getElementById('mm-img-input').click();
}
function mmTriggerVideoInsert(){
  if(!mmSelectedNode){ alert('Chọn một node trước, sau đó bấm 🎬 Video.'); return; }
  mmPendingMediaNodeId=mmSelectedNode;
  document.getElementById('mm-vid-input').click();
}
function mmHandleImageFile(e){
  const file=e.target.files[0]; if(!file) return;
  const reader=new FileReader();
  reader.onload=ev=>{
    const id=mmPendingMediaNodeId; if(!mmData.nodes[id]) return;
    if(!mmData.nodes[id].media) mmData.nodes[id].media=[];
    mmData.nodes[id].media.push({type:'image',src:ev.target.result});
    mmSaveState(); mmRenderAll();
  };
  reader.readAsDataURL(file); e.target.value='';
}
function mmHandleVideoFile(e){
  const file=e.target.files[0]; if(!file) return;
  const url=URL.createObjectURL(file);
  const id=mmPendingMediaNodeId; if(!mmData.nodes[id]) return;
  if(!mmData.nodes[id].media) mmData.nodes[id].media=[];
  mmData.nodes[id].media.push({type:'video',src:url});
  mmSaveState(); mmRenderAll(); e.target.value='';
}

/* ── MM Context menu ── */
let mmCtxEl=null;
function mmShowCtx(e,nodeId){
  if(mmCtxEl){ mmCtxEl.remove(); mmCtxEl=null; }
  const menu=document.createElement('div');
  menu.style.cssText=`position:fixed;left:${e.clientX}px;top:${e.clientY}px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:6px;z-index:9999;box-shadow:0 8px 24px rgba(0,0,0,.3);min-width:160px`;
  const items=[
    ['Tab → Node con','➕',()=>{ const c=mmAddChildNode(nodeId); if(c) mmEditNode(c); }],
    ['Node cùng cấp','↩',()=>{ const s=mmAddSiblingNode(nodeId); if(s) mmEditNode(s); }],
    ['Sửa văn bản','✏️',()=>mmEditNode(nodeId)],
    ['🖼 Chèn ảnh','',()=>{ mmSelectedNode=nodeId; mmTriggerImageInsert(); }],
    ['🎬 Chèn video','',()=>{ mmSelectedNode=nodeId; mmTriggerVideoInsert(); }],
    ['Thu/Mở nhánh','⤡',()=>mmToggleCollapse(nodeId)],
    ['Xóa node','🗑',()=>mmDeleteNode(nodeId)],
  ];
  items.forEach(([label,icon,fn])=>{
    const btn=document.createElement('div');
    btn.style.cssText='padding:7px 12px;cursor:pointer;border-radius:6px;font-size:13px;display:flex;gap:8px;align-items:center';
    btn.innerHTML=`<span>${icon}</span><span>${label}</span>`;
    btn.onmouseenter=()=>btn.style.background='var(--surface2)';
    btn.onmouseleave=()=>btn.style.background='';
    btn.onclick=()=>{ fn(); menu.remove(); mmCtxEl=null; };
    menu.appendChild(btn);
  });
  document.body.appendChild(menu);
  mmCtxEl=menu;
  const close=e=>{ if(!menu.contains(e.target)){ menu.remove(); mmCtxEl=null; document.removeEventListener('mousedown',close); } };
  setTimeout(()=>document.addEventListener('mousedown',close),0);
}

/* ─────────────────────────────────────────────────
   SECTION 10: INIT
───────────────────────────────────────────────── */
buildTree();
// Open first module by default
if(MODULES[0]&&MODULES[0].detail) selectDetail(MODULES[0]);