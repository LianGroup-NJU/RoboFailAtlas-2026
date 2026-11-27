(() => {
  'use strict';
  const host=document.querySelector('.method-walkthrough');
  if(!host)return;
  const buttons=[...host.querySelectorAll('[data-method-step]')];
  const overlay=host.querySelector('.method-overlay'), description=host.querySelector('.method-description');
  const play=host.querySelector('.method-play');
  let steps=[],index=0,timer=null;
  const stop=()=>{clearInterval(timer);timer=null;play.textContent='Play walkthrough';play.setAttribute('aria-pressed','false');};
  function show(i){
    index=i;
    buttons.forEach((button,k)=>button.setAttribute('aria-pressed',String(k===i)));
    overlay.replaceChildren(...steps[i].boxes.map(([x,y,w,h])=>{
      const box=document.createElement('span');box.className='method-highlight';
      Object.assign(box.style,{left:`${x*100}%`,top:`${y*100}%`,width:`${w*100}%`,height:`${h*100}%`});return box;
    }));
    description.textContent=steps[i].text;
  }
  fetch('assets/method-steps.json').then(response=>{if(!response.ok)throw new Error('Method steps unavailable');return response.json();}).then(data=>{
    steps=data;show(0);buttons.forEach((button,i)=>{button.disabled=false;button.addEventListener('click',()=>{stop();show(i);});});
    play.disabled=false;play.addEventListener('click',()=>{
      if(timer){stop();return;}
      if(index===steps.length-1)show(0);
      play.textContent='Pause walkthrough';play.setAttribute('aria-pressed','true');
      timer=setInterval(()=>{if(index===steps.length-1){stop();return;}show(index+1);},2400);
    });
    new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe(host);
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  }).catch(()=>{play.hidden=true;buttons.forEach(button=>button.hidden=true);description.textContent='Joint task and failure prediction supplies frozen reward feedback during PPO.';});
})();
