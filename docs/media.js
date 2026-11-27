(() => {
  'use strict';
  // Complete local Blob sources allow seeking even when a host ignores HTTP Range.
  const states=new Map();
  window.loadSeekableVideo=async(video,url)=>{
    const previous=states.get(video);
    if(previous?.url===url)return previous.promise;
    if(previous){previous.controller.abort();if(previous.blob)URL.revokeObjectURL(previous.blob);}
    video.pause();video.removeAttribute('src');video.load();
    video.dataset.mediaReady='false';video.dataset.source=url;
    const state={url,controller:new AbortController(),blob:null,promise:null};states.set(video,state);
    state.promise=(async()=>{
      try{
        const response=await fetch(url,{credentials:'same-origin',signal:state.controller.signal});
        if(!response.ok)throw new Error(`Media HTTP ${response.status}`);
        const blob=await response.blob();
        if(states.get(video)!==state)return;
        state.blob=URL.createObjectURL(new Blob([blob],{type:'video/mp4'}));
        const ready=new Promise((resolve,reject)=>{
          video.addEventListener('loadedmetadata',resolve,{once:true});
          video.addEventListener('error',reject,{once:true});
        });
        video.src=state.blob;video.preload='auto';video.load();await ready;
        if(states.get(video)===state){video.dataset.mediaReady='true';video.dispatchEvent(new Event('seekableready'));}
      }catch(error){
        if(error.name==='AbortError'||states.get(video)!==state)return;
        video.dataset.mediaReady='fallback';video.src=url;video.load();
      }
    })();
    return state.promise;
  };
  const stamp=t=>`${Math.floor(Math.max(0,t||0)/60).toString().padStart(2,'0')}:${Math.floor(Math.max(0,t||0)%60).toString().padStart(2,'0')}`;
  document.querySelectorAll('[data-player]').forEach(controls=>{
    const video=document.getElementById(controls.dataset.player),slider=controls.querySelector('input'),time=controls.querySelector('output');
    let dragging=false;
    function update(){
      const duration=Number.isFinite(video.duration)?video.duration:0;
      slider.disabled=duration<=0;slider.max=String(duration||1);
      if(!dragging)slider.value=String(video.currentTime||0);
      time.value=`${stamp(video.currentTime)} / ${stamp(duration)}`;
    }
    function commit(){if(Number.isFinite(video.duration))video.currentTime=Math.min(Number(slider.value),Math.max(0,video.duration-.02));}
    slider.addEventListener('pointerdown',()=>{dragging=true;});
    slider.addEventListener('input',commit);
    slider.addEventListener('change',()=>{commit();dragging=false;update();});
    slider.addEventListener('pointerup',()=>{commit();dragging=false;});
    slider.addEventListener('pointercancel',()=>{dragging=false;update();});
    for(const event of ['loadedmetadata','durationchange','timeupdate','seeked','emptied'])video.addEventListener(event,update);
    update();
  });
  const hero=document.getElementById('hero-video');if(hero?.dataset.mediaSrc)window.loadSeekableVideo(hero,hero.dataset.mediaSrc);
  const observer=new IntersectionObserver(entries=>{
    for(const entry of entries)if(entry.isIntersecting){
      const video=entry.target;window.loadSeekableVideo(video,video.dataset.mediaSrc);observer.unobserve(video);
    }
  },{rootMargin:'600px'});
  document.querySelectorAll('video[data-media-src]').forEach(video=>{if(video!==hero)observer.observe(video);});
  window.addEventListener('pagehide',event=>{if(event.persisted)return;for(const state of states.values()){state.controller.abort();if(state.blob)URL.revokeObjectURL(state.blob);}});
})();
