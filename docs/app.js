(() => {
  'use strict';
  const all = window.FAILATLAS_CASES || [];
  const $ = id => document.getElementById(id);
  let filtered = [...all], current = null;
  const video = $('case-video');
  const recoveryNames = {recovered:'Recovered',attempted_not_recovered:'Attempted, not recovered',not_recovered_as_of_checkpoint:'Not recovered as of checkpoint'};
  const humanTask = s => s.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
  const labelFragment = (labels, color) => {
    const frag = document.createDocumentFragment();
    for (const label of labels) {
      const item = document.createElement('div');
      const name = document.createElement('span'); name.className = `tag ${color}`; name.textContent = label.name;
      const code = document.createElement('small'); code.className='code-label'; code.textContent=`${label.code} · ${label.status}`;
      item.append(name, code); frag.append(item);
    }
    if (!labels.length) { const item = document.createElement('span'); item.textContent='Not assigned'; frag.append(item); }
    return frag;
  };
  function seek(frame) {
    if (!current) return;
    const value = Math.max(0, Math.min(current.frame_count - 1, frame));
    video.pause();
    const apply = () => { video.currentTime = Math.min(value / current.fps, Math.max(0, (video.duration || Infinity) - 0.002)); updateFrame(value); };
    if (video.readyState >= 1) apply(); else video.addEventListener('loadedmetadata', apply, {once:true});
  }
  function updateFrame(override) {
    if (!current) return;
    const frame = typeof override === 'number' ? override : Math.min(current.frame_count - 1, Math.floor(video.currentTime * current.fps + 0.001));
    $('frame-display').textContent=`Frame ${frame}`; $('frame-slider').value=String(frame);
    const segment = current.segments.find(s=>frame>=s.start_frame && (frame<s.end_frame || frame===current.frame_count-1 && frame===s.end_frame));
    $('current-action').textContent=segment ? `${segment.atom_code.replace('ATOM.','')} (${segment.arm})` : 'Unannotated interval';
    document.querySelectorAll('#action-timeline button').forEach(b=>b.classList.toggle('active', Boolean(segment && b.dataset.segment===segment.id)));
  }
  function choose(id) {
    current = all.find(c=>c.id===id); if (!current) return;
    video.pause(); video.poster=current.frames[0].image;
    $('frame-slider').disabled=true;
    const chosenId=current.id;
    window.loadSeekableVideo(video,current.video).then(()=>{if(current.id===chosenId&&video.readyState>=1){$('frame-slider').disabled=false;video.currentTime=0;updateFrame(0);}});
    video.playbackRate=Number($('playback-speed').value);
    $('case-select').value=id;
    $('case-position').textContent=`${filtered.findIndex(c=>c.id===id)+1} / ${filtered.length}`;
    $('case-title').textContent=current.title; $('case-task').textContent=humanTask(current.task);
    $('case-instruction').textContent=current.instruction;
    $('type-labels').replaceChildren(labelFragment(current.type_labels,'coral'));
    $('outcome-labels').replaceChildren(labelFragment(current.outcome_labels,'neutral'));
    const recovery = document.createElement('span');recovery.className=`tag ${current.recovery==='recovered'?'mint':'neutral'}`;recovery.textContent=recoveryNames[current.recovery]||current.recovery;
    $('recovery-label').replaceChildren(recovery);
    $('terminal-label').textContent=`T = ${current.terminal_success}`;
    $('event-meta').textContent=`Event f${current.event.start_frame}–f${current.event.end_frame} · checkpoint f${current.event.checkpoint_frame ?? 'unknown'}`;
    $('frame-slider').max=String(current.frame_count-1); $('frame-total').textContent=`/ ${current.frame_count-1}`;
    $('action-timeline').replaceChildren();
    current.segments.forEach(s=>{
      const b=document.createElement('button');b.type='button';b.dataset.segment=s.id;
      const width=100*(s.end_frame-s.start_frame)/Math.max(1,current.frame_count-1);
      b.style.left=`${100*s.start_frame/Math.max(1,current.frame_count-1)}%`;b.style.width=`${width}%`;
      b.title=`${s.atom_code} (${s.arm}), frames ${s.start_frame}–${s.end_frame}`; b.setAttribute('aria-label',`Seek to ${b.title}`);
      b.textContent=width>8?s.atom_code.replace('ATOM.',''):'';b.addEventListener('click',()=>seek(s.start_frame));$('action-timeline').append(b);
    });
    $('stage-grid').replaceChildren();
    current.frames.forEach(f=>{
      const b=document.createElement('button');b.className='stage-button';b.type='button';b.setAttribute('aria-label',`Seek to ${f.caption}, frame ${f.frame}`);
      const img=document.createElement('img');img.src=f.image;img.alt=`${current.title}: ${f.caption}, original frame ${f.frame}`;img.width=320;img.height=240;
      const text=document.createElement('span');text.textContent=f.caption;const frame=document.createElement('small');frame.textContent=`Original frame ${f.frame}`;
      b.append(img,text,frame);b.addEventListener('click',()=>seek(f.frame));$('stage-grid').append(b);
    });
    $('case-json').textContent=JSON.stringify(current,null,2); updateFrame(0);
  }
  function filter(category) {
    filtered=all.filter(c=>category==='all'||c.category===category);
    $('case-select').replaceChildren(...filtered.map(c=>{const o=document.createElement('option');o.value=c.id;o.textContent=`${humanTask(c.task)} · ${c.title}`;return o;}));
    document.querySelectorAll('.filter').forEach(b=>{const active=b.dataset.category===category;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
    if (filtered.length) choose(filtered.some(c=>c.id===current?.id)?current.id:filtered[0].id);
  }
  document.querySelectorAll('.filter').forEach(b=>b.addEventListener('click',()=>filter(b.dataset.category)));
  $('case-select').addEventListener('change',e=>choose(e.target.value));
  for (const [id,delta] of [['previous-case',-1],['next-case',1]]) $(id).addEventListener('click',()=>{const i=filtered.findIndex(c=>c.id===current?.id);choose(filtered[(i+delta+filtered.length)%filtered.length].id);});
  $('frame-slider').addEventListener('input',e=>seek(Number(e.target.value)));
  $('playback-speed').addEventListener('change',e=>{video.playbackRate=Number(e.target.value);});
  $('jump-event').addEventListener('click',()=>seek(current.event.start_frame));
  video.addEventListener('timeupdate',()=>updateFrame());video.addEventListener('seeked',()=>updateFrame());video.addEventListener('loadedmetadata',()=>{$('frame-slider').disabled=false;video.playbackRate=Number($('playback-speed').value);updateFrame();});
  video.addEventListener('error',()=>{const e=$('demo-error');e.hidden=false;e.textContent='The selected video could not be loaded. The keyframes and JSON remain available.';});
  video.addEventListener('loadeddata',()=>{$('demo-error').hidden=true;});
  $('download-json').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(current,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`${current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  if (all.length) {current=all.find(c=>c.id==='recovery_states__attempted_not_recovered__02');filter('all');} else {$('demo-error').hidden=false;$('demo-error').textContent='Example data is unavailable.';}
  fetch('release.json').then(r=>r.json()).then(config=>{
    document.querySelectorAll('[data-resource]').forEach(a=>{const value=config[a.dataset.resource];if(value&&/^https:\/\//.test(value)){a.href=value;a.removeAttribute('aria-disabled');const note=a.querySelector('small');if(note)note.textContent='View resource';}});
    if(config.authors?.length) $('authors').textContent=config.authors.join(' · ');
    if(config.affiliations?.length){$('affiliations').hidden=false;$('affiliations').textContent=config.affiliations.join(' · ');}
    if(config.bibtex){$('citation-section').hidden=false;$('bibtex').textContent=config.bibtex;$('copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(config.bibtex);$('copy-citation').textContent='Copied';}catch{$('copy-citation').textContent='Select the citation to copy';}});}
  }).catch(()=>{});
})();
