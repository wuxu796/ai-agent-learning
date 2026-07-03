/* 食鉴 2.0 — 前端交互逻辑 */

// ─── DOM 引用 ───
const $id = id => document.getElementById(id);
const step1 = $id('step1'), step2 = $id('step2'), step3 = $id('step3'), step4 = $id('step4');
const fileInput = $id('fileInput');
const captureCard = $id('captureCard');
const previewImg = $id('previewImg');
const previewPlaceholder = $id('previewPlaceholder');
const analyzeBtn = $id('analyzeBtn');
const progressFill = $id('progressFill');
const load1 = $id('load1'), load2 = $id('load2'), load3 = $id('load3');
const loadingBox = $id('loadingBox');
const resultBox = $id('resultBox');
const errorBanner = $id('errorBanner');
const expandBtn = $id('expandBtn');
const ocrEditor = $id('ocrEditor');
const confirmOcrBtn = $id('confirmOcrBtn');
const editBackBtn = $id('editBackBtn');
const ocrInputs = {
  product_name: $id('ocrProductName'),
  ingredients: $id('ocrIngredients'),
  additives: $id('ocrAdditives'),
  energy_kj: $id('ocrEnergy'),
  fat_g: $id('ocrFat'),
  sat_fat_g: $id('ocrSatFat'),
  trans_fat_g: $id('ocrTransFat'),
  carb_g: $id('ocrCarb'),
  sugar_g: $id('ocrSugar'),
  protein_g: $id('ocrProtein'),
  sodium_mg: $id('ocrSodium'),
  serving_g: $id('ocrServing'),
};
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/jpg', 'image/pjpeg', 'image/png', 'image/webp'];
const GENERIC_UPLOAD_TYPES = ['application/octet-stream', 'binary/octet-stream'];
const ALLOWED_IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.jfif', '.png', '.webp'];
const REQUEST_TIMEOUT_MS = 120000;

let imageFile = null;
let visionResult = null;
let selectedHealth = [];
let selectedFocus = [];
let chatSessionId = null;

// ─── 功能菜单切换 ───
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode;
    document.querySelectorAll('.mode-btn').forEach(item => {
      const active = item.dataset.mode === mode;
      item.classList.toggle('active', active);
      item.setAttribute('aria-expanded', active ? 'true' : 'false');
    });
    document.querySelectorAll('.mode-panel').forEach(panel => {
      panel.classList.toggle('active', panel.id === mode + 'Panel');
    });
    hideError();
  });
});

// ─── 步骤切换 ───
function showStep(n) {
  [step1, step2, step3, step4].forEach((s, i) => s.classList.toggle('active', i + 1 === n));
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── 错误提示 ───
function showError(msg) {
  if (!errorBanner) return;
  errorBanner.textContent = '';
  const title = document.createElement('span');
  title.className = 'error-title';
  title.textContent = '出错了';
  errorBanner.appendChild(title);
  errorBanner.appendChild(document.createTextNode(String(msg || '未知错误')));
  errorBanner.style.display = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hideError() {
  if (errorBanner) errorBanner.style.display = 'none';
}

// ─── 步骤①：拍照 ───
captureCard.addEventListener('click', () => fileInput.click());
captureCard.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) {
    if (!isValidImageFile(file)) {
      imageFile = null;
      fileInput.value = '';
      return;
    }
    imageFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.src = ev.target.result;
      previewImg.style.display = 'block';
      previewPlaceholder.style.display = 'none';
      showStep(2);
    };
    reader.readAsDataURL(imageFile);
  }
});

// ─── 步骤②：确认 ───
$id('retakeBtn').addEventListener('click', () => { imageFile = null; fileInput.value = ''; showStep(1); });
$id('confirmBtn').addEventListener('click', () => showStep(3));

// ─── 步骤③：BMI ───
const heightInput = $id('heightInput'), weightInput = $id('weightInput');
const bmiValue = $id('bmiValue'), bmiStatus = $id('bmiStatus'), bmiTip = $id('bmiTip');

function updateBMI() {
  const h = parseFloat(heightInput.value) || 170;
  const w = parseFloat(weightInput.value) || 65;
  const bmi = w / ((h / 100) ** 2);
  bmiValue.textContent = bmi.toFixed(1);
  if (bmi < 18.5) { bmiStatus.textContent = '🔵 偏瘦'; bmiTip.textContent = '可以适当增重，多吃蛋白质和优质碳水'; }
  else if (bmi < 24) { bmiStatus.textContent = '🟢 正常'; bmiTip.textContent = '体重在健康范围，继续保持'; }
  else if (bmi < 28) { bmiStatus.textContent = '⚠️ 超重'; bmiTip.textContent = '超重，建议关注热量摄入'; }
  else { bmiStatus.textContent = '🔴 肥胖'; bmiTip.textContent = '肥胖，建议制定系统减脂计划'; }
}
heightInput.addEventListener('input', updateBMI);
weightInput.addEventListener('input', updateBMI);
updateBMI();

// ─── 标签切换 ───
document.querySelectorAll('.tag-group').forEach(group => {
  group.addEventListener('click', e => {
    const tag = e.target.closest('.tag');
    if (!tag) return;
    tag.classList.toggle('selected');
    tag.setAttribute('aria-checked', tag.classList.contains('selected') ? 'true' : 'false');
    selectedHealth = [...document.querySelectorAll('#healthTags .tag.selected')].map(t => t.dataset.val);
    selectedFocus = [...document.querySelectorAll('#focusTags .tag.selected')].map(t => t.dataset.val);
  });
  // 键盘支持
  group.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const tag = e.target.closest('.tag');
      if (!tag) return;
      tag.click();
    }
  });
});

// ─── 构建用户画像（匹配后端 profile 结构） ───
function getProfile() {
  const h = parseFloat(heightInput.value) || null;
  const w = parseFloat(weightInput.value) || null;
  const age = parseFloat($id('ageInput').value) || null;
  const gender = $id('genderSelect').value || null;
  const activity = $id('activitySelect').value || null;

  // 合并 concerns + focus → 后端只认 concerns，focus 作为额外关注点合并
  const concerns = [...new Set([...selectedHealth, ...selectedFocus])];

  const profile = {};
  if (h) profile.height = h;
  if (w) profile.weight = w;
  if (age) profile.age = age;
  if (gender) profile.gender = gender;
  if (activity) profile.activity = activity;
  if (concerns.length) profile.concerns = concerns;

  return Object.keys(profile).length ? profile : null;
}

// ─── 分析入口 ───
if (analyzeBtn) {
  analyzeBtn.addEventListener('click', runAnalysis);
}
if (confirmOcrBtn) {
  confirmOcrBtn.addEventListener('click', runNutritionAnalysis);
}
if (editBackBtn) {
  editBackBtn.addEventListener('click', () => {
    if (ocrEditor) ocrEditor.style.display = 'none';
    showStep(3);
  });
}

async function runAnalysis() {
  if (!analyzeBtn || analyzeBtn.disabled) return;
  if (!imageFile) { showError('请先拍照上传配料表图片'); showStep(1); return; }

  hideError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = '⏳ 识别中...';
  showStep(4);
  resetLoading();

  try {
    // ① OCR 识别
    updateLoading(1, 10);
    const vForm = new FormData();
    vForm.append('image', imageFile);
    const vResp = await apiFetch('/vision', { method: 'POST', body: vForm }, '配料表识别');
    if (!vResp.ok) throw new Error(await readApiError(vResp, '配料表识别失败'));
    visionResult = await vResp.json();
    updateLoading(1, 33, true);
    renderOcrEditor(visionResult);
  } catch (err) {
    console.error('[食鉴] 分析失败:', err);
    showError(err.message);
    // 终极兜底：如果 errorBanner 不可用，至少 alert 出来
    if (!errorBanner) { alert('分析失败：' + err.message); }
    showStep(3); // 回到步骤③让用户修改，错误提示全局可见不会被抹掉
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 开始分析';
  }
}

async function runNutritionAnalysis() {
  if (!confirmOcrBtn || confirmOcrBtn.disabled) return;
  if (!visionResult) { showError('请先识别配料表'); showStep(3); return; }

  hideError();
  confirmOcrBtn.disabled = true;
  confirmOcrBtn.textContent = '⏳ 分析中...';
  resetLoading();
  showStep(4);
  updateLoading(1, 33, true);

  try {
    visionResult = buildEditedVisionJson();

    updateLoading(2, 50);
    await sleep(200);
    updateLoading(2, 66, true);
    updateLoading(3, 75);

    const aResp = await apiFetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vision_json: visionResult, profile: getProfile() }),
    }, '营养分析');
    if (!aResp.ok) throw new Error(await readApiError(aResp, '营养分析失败'));
    const result = await aResp.json();

    updateLoading(3, 100, true);
    await sleep(300);
    renderResult(result);
  } catch (err) {
    console.error('[食鉴] 分析失败:', err);
    showError(err.message);
    if (ocrEditor) ocrEditor.style.display = 'block';
    loadingBox.style.display = 'none';
  } finally {
    confirmOcrBtn.disabled = false;
    confirmOcrBtn.textContent = '确认并分析';
  }
}

function resetLoading() {
  loadingBox.style.display = 'block';
  resultBox.style.display = 'none';
  if (ocrEditor) ocrEditor.style.display = 'none';
  hideError();
  progressFill.style.width = '0%';
  [load1, load2, load3].forEach(el => {
    el.classList.remove('done');
    el.textContent = el.textContent.replace(/^✅/, '⏳').replace(/ ✓$/, '');
  });
}

function renderOcrEditor(data) {
  loadingBox.style.display = 'none';
  resultBox.style.display = 'none';
  if (!ocrEditor) return;

  const nutrition = data?.nutrition || {};
  const per100g = nutrition.per_100g || {};

  ocrInputs.product_name.value = data?.product_name || '';
  ocrInputs.ingredients.value = listToLines(data?.ingredients || []);
  ocrInputs.additives.value = listToLines(data?.additives || []);
  setNumberInput(ocrInputs.energy_kj, per100g.energy_kj);
  setNumberInput(ocrInputs.fat_g, per100g.fat_g);
  setNumberInput(ocrInputs.sat_fat_g, per100g.sat_fat_g);
  setNumberInput(ocrInputs.trans_fat_g, per100g.trans_fat_g);
  setNumberInput(ocrInputs.carb_g, per100g.carb_g);
  setNumberInput(ocrInputs.sugar_g, per100g.sugar_g);
  setNumberInput(ocrInputs.protein_g, per100g.protein_g);
  setNumberInput(ocrInputs.sodium_mg, per100g.sodium_mg);
  setNumberInput(ocrInputs.serving_g, nutrition.serving_g);

  ocrEditor.style.display = 'block';
  ocrInputs.product_name.focus();
}

function buildEditedVisionJson() {
  return {
    product_name: ocrInputs.product_name.value.trim(),
    ingredients: linesToList(ocrInputs.ingredients.value),
    additives: linesToList(ocrInputs.additives.value),
    nutrition: {
      per_100g: {
        energy_kj: readNullableNumber(ocrInputs.energy_kj),
        fat_g: readNullableNumber(ocrInputs.fat_g),
        sat_fat_g: readNullableNumber(ocrInputs.sat_fat_g),
        trans_fat_g: readNullableNumber(ocrInputs.trans_fat_g),
        carb_g: readNullableNumber(ocrInputs.carb_g),
        sugar_g: readNullableNumber(ocrInputs.sugar_g),
        protein_g: readNullableNumber(ocrInputs.protein_g),
        sodium_mg: readNullableNumber(ocrInputs.sodium_mg),
      },
      serving_g: readNullableNumber(ocrInputs.serving_g),
    },
  };
}

function listToLines(items) {
  return (Array.isArray(items) ? items : []).filter(Boolean).join('\n');
}

function linesToList(text) {
  return String(text || '')
    .split(/\n+/)
    .map(item => item.trim())
    .filter(Boolean);
}

function setNumberInput(input, value) {
  input.value = value === null || value === undefined ? '' : String(value);
}

function readNullableNumber(input) {
  const raw = String(input.value || '').trim();
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error('营养成分数字不能为负数或非数字');
  }
  return value;
}

function updateLoading(step, pct, done) {
  if (done) {
    const el = step === 1 ? load1 : step === 2 ? load2 : load3;
    el.classList.add('done');
    el.textContent = el.textContent.replace('⏳', '✅') + ' ✓';
  }
  progressFill.style.width = pct + '%';
}

// ─── Markdown 解析（安全的块级处理，避免 <p> 包裹标题） ───
function parseMarkdown(text) {
  if (!text) return '';

  // 拆分为块（按空行分隔）
  const blocks = String(text).split(/\n\n+/);
  return blocks.map(block => {
    block = block.trim();
    if (!block) return '';

    // 标题
    if (/^###\s/.test(block)) {
      return '<h3>' + processInline(block.slice(4)) + '</h3>';
    }
    if (/^##\s/.test(block)) {
      return '<h2>' + processInline(block.slice(3)) + '</h2>';
    }

    // 无序列表
    if (/^[-*]\s/.test(block)) {
      const items = block.split(/\n/).filter(l => /^[-*]\s/.test(l));
      return '<ul>' + items.map(item =>
        '<li>' + processInline(item.replace(/^[-*]\s+/, '')) + '</li>'
      ).join('') + '</ul>';
    }

    // 普通段落
    let html = processInline(block);
    html = html.replace(/\n/g, '<br>');
    return '<p>' + html + '</p>';
  }).join('\n');
}

function processInline(text) {
  // 先转义，再允许非常有限的 Markdown 加粗语法。
  return escapeHtml(String(text)).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

// ─── 结果渲染 ───
function renderResult(r) {
  loadingBox.style.display = 'none';
  if (ocrEditor) ocrEditor.style.display = 'none';
  resultBox.style.display = 'block';

  renderProduct(r);
  renderConclusion(r);
  renderIndicators(r);
  renderIngredients(r);
  renderAdvice(r);
  renderAnalysis(r);
}

// ── 品名 ──
function renderProduct(r) {
  const name = r.product?.name || visionResult?.product_name || '未知产品';
  const summary = r.product?.summary || '';
  $id('productBox').innerHTML =
    `<div class="name">📦 品名：${escapeHtml(name)}</div>` +
    (summary ? `<div class="summary">${escapeHtml(summary)}</div>` : '') +
    `<div class="ocr-note">以上为 OCR 识别结果，仅供参考</div>`;
}

// ── 结论卡片 ──
function renderConclusion(r) {
  const rawRisk = r.verdict?.risk_level || 'medium';
  const conclMap = {
    low:    ['✅', '放心食用'],
    medium: ['⚠️', '建议少量食用'],
    high:   ['🔴', '不建议食用'],
  };
  const risk = Object.prototype.hasOwnProperty.call(conclMap, rawRisk) ? rawRisk : 'medium';
  const [cIcon, cText] = conclMap[risk] || conclMap.medium;
  const card = $id('conclusionCard');
  card.className = `conclusion-card ${risk}`;
  card.innerHTML = `<span class="c-icon">${cIcon}</span>${cText}`;
}

// ── 关键指标（6 项红绿灯） ──
function renderIndicators(r) {
  const tl = r.verdict?.traffic_lights || {};
  const per100g = r.nutrition_detail?.per_100g || {};

  const indicatorDefs = [
    { key: 'fat',      icon: '🧈', label: '脂肪',   unit: 'g',  val: per100g.fat_g },
    { key: 'sugar',    icon: '🍬', label: '含糖量', unit: 'g',  val: per100g.sugar_g },
    { key: 'sodium',   icon: '🧂', label: '钠',     unit: 'mg', val: per100g.sodium_mg },
    { key: 'protein',   icon: '🥚', label: '蛋白质',   unit: 'g',  val: per100g.protein_g },
    { key: 'trans_fat', icon: '⚠️', label: '反式脂肪', unit: 'g',  val: per100g.trans_fat_g },
    { key: 'additives', icon: '🧪', label: '添加剂',   unit: '项', val: (r.ingredients || []).length },
  ];

  $id('indicatorRow').innerHTML = indicatorDefs.map(def => {
    const level = ['red', 'yellow', 'green'].includes(tl[def.key]) ? tl[def.key] : 'green';
    const valText = def.val != null ? def.val + def.unit : '?';
    return `<div class="indicator-card">
      <span class="i-icon">${def.icon}</span>
      <div class="i-value">${escapeHtml(valText)}</div>
      <div class="i-label">${def.label}</div>
      <span class="i-dot ${level}" aria-label="${level === 'red' ? '高' : level === 'yellow' ? '中' : '低'}"></span>
    </div>`;
  }).join('');
}

// ── 配料红绿灯（高风险排上面） ──
function renderIngredients(r) {
  const ings = r.ingredients || [];
  const order = { high: 0, medium: 1, safe: 2 };
  ings.sort((a, b) => (order[a.risk] || 2) - (order[b.risk] || 2));
  const tagMap = { high: ['高风险', 'high'], medium: ['注意', 'medium'], safe: ['安全', 'safe'] };

  const visible = 4;
  const list = $id('trafficList');
  list.innerHTML = ings.map((ing, i) => {
    const [label, cls] = tagMap[ing.risk] || tagMap.safe;
    const hidden = i >= visible ? ' hidden-traffic' : '';
    return `<li class="${hidden}"><span class="traffic-tag ${cls}">${label}</span>${escapeHtml(ing.name)}</li>`;
  }).join('');

  const count = ings.length;
  if (count > visible) {
    expandBtn.style.display = 'block';
    let expanded = false;
    expandBtn.onclick = () => {
      expanded = !expanded;
      list.querySelectorAll('.hidden-traffic').forEach(el => el.classList.toggle('show', expanded));
      expandBtn.textContent = expanded ? '收起 ▲' : `展开全部（${count - visible}项） ▼`;
    };
    expandBtn.textContent = `展开全部（${count - visible}项） ▼`;
  } else {
    expandBtn.style.display = 'none';
  }
}

// ── 个性化建议 ──
function renderAdvice(r) {
  const advice = r.personalized?.advice || '分析完成，请根据配料信息合理饮食。';
  $id('adviceBody').innerHTML = parseMarkdown(advice);
}

// ── 营养师解读（配料 + 营养成分 LLM 分析） ──
function renderAnalysis(r) {
  const ingAnalysis = r.explanation?.ingredient_analysis || '';
  const nutAnalysis = r.explanation?.nutrition_analysis || '';

  const parts = [];
  if (ingAnalysis) parts.push('<h3>🔬 配料解读</h3>' + parseMarkdown(ingAnalysis));
  if (nutAnalysis && nutAnalysis !== ingAnalysis) {
    parts.push('<h3>📊 营养分析</h3>' + parseMarkdown(nutAnalysis));
  }

  $id('analysisBody').innerHTML = parts.join('') ||
    '<p>AI 解读暂无内容，请查看上方规则分析结果。</p>';
}

// ─── 工具函数 ───
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function isValidImageFile(file) {
  const name = (file.name || '').toLowerCase();
  const suffixOk = ALLOWED_IMAGE_SUFFIXES.some(suffix => name.endsWith(suffix));
  const genericType = file.type && GENERIC_UPLOAD_TYPES.includes(file.type);
  const typeOk = ALLOWED_IMAGE_TYPES.includes(file.type) || ((!file.type || genericType) && suffixOk);

  if (!typeOk) {
    showError('仅支持 JPG、JPEG、PNG、WEBP 图片');
    return false;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showError('图片过大，请上传 8MB 以内的图片');
    return false;
  }
  return true;
}

// ─── 重新测试 ───
$id('retryBtn').addEventListener('click', () => {
  imageFile = null;
  visionResult = null;
  selectedHealth = [];
  selectedFocus = [];
  fileInput.value = '';
  if (ocrEditor) ocrEditor.style.display = 'none';
  document.querySelectorAll('.tag.selected').forEach(t => {
    t.classList.remove('selected');
    t.setAttribute('aria-checked', 'false');
  });
  showStep(1);
});

// ─── 营养问答 ───
const chatForm = $id('chatForm');
const chatInput = $id('chatInput');
const chatLog = $id('chatLog');

if (chatForm) {
  chatForm.addEventListener('submit', async e => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    appendChatMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = '';

    const pending = appendChatMessage('assistant', '正在查知识库...');
    const sendBtn = chatForm.querySelector('button');
    sendBtn.disabled = true;

    try {
      if (!chatSessionId) {
        const sessionResp = await apiFetch('/api/sessions', { method: 'POST' }, '创建会话');
        if (!sessionResp.ok) throw new Error(await readApiError(sessionResp, '创建会话失败'));
        const session = await sessionResp.json();
        chatSessionId = session.session_id;
      }

      const resp = await apiFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: chatSessionId, message: text }),
      }, '问答');
      if (!resp.ok) throw new Error(await readApiError(resp, '问答失败'));
      const data = await resp.json();
      pending.querySelector('.bubble').innerHTML = parseMarkdown(data.reply || '没有返回内容');
    } catch (err) {
      pending.querySelector('.bubble').textContent = err.message;
    } finally {
      sendBtn.disabled = false;
      chatInput.focus();
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  });

  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
  });
}

function appendChatMessage(role, text) {
  const row = document.createElement('div');
  row.className = `chat-message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'assistant') {
    bubble.innerHTML = parseMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);
  chatLog.appendChild(row);
  chatLog.scrollTop = chatLog.scrollHeight;
  return row;
}

async function readApiError(resp, fallback) {
  try {
    const data = await resp.json();
    if (data?.detail?.message) return data.detail.message;
    if (typeof data?.detail === 'string') return data.detail;
  } catch (err) {
    // ignore parse errors and fall through to status text
  }
  return fallback + '（' + resp.status + '）';
}

async function apiFetch(url, options, label) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(`${label}超时，请稍后重试`);
    }
    throw new Error(
      `${label}连接失败，请确认不是直接打开本地 HTML，而是通过 Docker 服务地址访问，例如 http://localhost:8000，然后强制刷新后重试`
    );
  } finally {
    clearTimeout(timer);
  }
}
