// Dependency-free JS/HTTP smoke checks using a small DOM stub.
// This verifies rendering logic and flows, not browser layout or CSS.
// Run while app.py is running in the same environment: node test_dashboard.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, 'dashboard.html'), 'utf8');
const code = html.split('<script>')[1].split('</script>')[0];
const elements = new Map();
const downloads = [];
const blobs = [];
class Element {
  constructor(id) { this.id=id;this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.dataset={};this.files=[];this.classList={toggle:()=>{}}; }
  addEventListener() {}
  click() { if(this.download) downloads.push(this.download); }
  set innerHTML(value) {
    this.html=value;
    for(const m of value.matchAll(/id="([^"]+)"/g)) if(!elements.has(m[1])) elements.set(m[1], new Element(m[1]));
  }
  get innerHTML() { return this.html||''; }
}
for(const tag of html.matchAll(/<[^>]+id="([^"]+)"[^>]*>/g)) {
  const e = new Element(tag[1]);
  const v = tag[0].match(/value="([^"]*)"/);
  if(v) e.value=v[1];
  elements.set(e.id,e);
}
Object.entries({source:'demo',trials:'10000',confidence:'0.99'}).forEach(([id,v])=>elements.get(id).value=v);
elements.get('methods').innerHTML = html.split('<section class="tab-panel" id="methods" hidden>')[1].split('</section>')[0];
const tabs=['overview','models','stress','accounts','methods'].map(name=>{const e=new Element(name+'-tab');e.dataset.tab=name;return e;});
const context=vm.createContext({console,Intl,JSON,Math,Number,String,Blob,
  URL:{createObjectURL:blob=> {blobs.push(blob);return 'blob:stub';},revokeObjectURL:()=>{}},
  setTimeout,
  fetch:(url,options)=>fetch('http://127.0.0.1:8765'+url,options),
  document:{getElementById:id=>elements.get(id),
    querySelectorAll:s=>s==='.tab-panel'?tabs.map(t=>elements.get(t.dataset.tab)):s==='[data-tab]'?tabs:[...elements.values()],
    querySelector:s=>({textContent:html.split('<style>')[1].split('</style>')[0]}),
    createElement:()=>new Element('new')}
});
vm.runInContext(code,context);
async function check() {
  for(let i=0;i<400;i++){
    if(elements.get('status').textContent.startsWith('Analysis complete'))break;
    await new Promise(resolve=>setTimeout(resolve,25));
  }
  assert.match(elements.get('status').textContent,/Analysis complete/);
  assert.match(elements.get('overview').innerHTML,/Portfolio exposure/);
  assert.match(elements.get('models').innerHTML,/CreditRisk\+/);
  assert.match(elements.get('histogram').innerHTML,/<svg/);
  assert.equal(vm.runInContext('result.overview.accounts',context),400);
  vm.runInContext("changeTab('models')",context);
  assert.equal(elements.get('models').hidden,false);
  elements.get('report').onclick();
  assert.match(downloads.at(-1),/^DealerRisk_report_/);
  const reportText = await blobs.at(-1).text();
  assert.match(reportText,/Model contract/);
  assert.match(reportText,/Input SHA-256/);
  if(process.env.DEALERRISK_REPORT_PATH) fs.writeFileSync(process.env.DEALERRISK_REPORT_PATH,reportText);
  elements.get('account-search').value='DEMO-0001';
  elements.get('account-search').oninput();
  assert.match(elements.get('account-table').innerHTML,/DEMO-0001/);
  assert.doesNotMatch(elements.get('account-table').innerHTML,/DEMO-0002/);
  await elements.get('export-accounts').onclick();
  assert.equal(downloads.at(-1),'borrower_loss_contributions.csv');
  elements.get('rho').value='0.20';vm.runInContext('dirty()',context);
  assert.match(elements.get('status').textContent,/Settings changed/);
  elements.get('source').value='upload';elements.get('source').onchange();
  assert.equal(vm.runInContext('result',context),null);
  elements.get('upload').files=[{name:'custom.csv',size:100,text:async()=> 'loan_id,ead,pd_12m,lgd,sector\n0001,12000,.08,.5,North\n0002,15000,.04,.6,South\n'}];
  await elements.get('upload').onchange();
  elements.get('trials').value='1000';
  await vm.runInContext('run()',context);
  assert.equal(vm.runInContext('result.overview.accounts',context),2);
  assert.match(elements.get('provenance').textContent,/custom.csv/);
  elements.get('upload').files=[{name:'invalid.csv',size:100,text:async()=> 'loan_id,ead,pd_12m,lgd\n1,100,12,.5'}];
  await elements.get('upload').onchange();await vm.runInContext('run()',context);
  assert.match(elements.get('status').textContent,/decimal/);
  assert.equal(vm.runInContext('result',context),null);
  console.log('PASS: automatic demo, tabs, histogram generation, report/CSV downloads, search, changed settings, uploaded CSV and invalid CSV flow.');
}
check().catch(error=>{console.error(error);process.exitCode=1;});
