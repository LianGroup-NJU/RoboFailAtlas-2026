(() => {
  const header=document.getElementById('top'),sticky=document.getElementById('sticky-header');
  new IntersectionObserver(([entry])=>sticky.classList.toggle('visible',!entry.isIntersecting),{threshold:0}).observe(header);
  const links=[...document.querySelectorAll('.contents a')];
  const targets=links.map(link=>document.querySelector(link.getAttribute('href'))).filter(Boolean);
  let scheduled=false;
  function update(){scheduled=false;let active=targets[0];for(const target of targets){if(target.getBoundingClientRect().top<140)active=target;}
    links.forEach(link=>{const current=link.getAttribute('href')==='#'+active.id;link.classList.toggle('active',current);if(current)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});}
  addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(update);}},{passive:true});update();
  const summary=document.querySelector('.video-collapse');summary.addEventListener('toggle',()=>{if(!summary.open)document.getElementById('overview-video').pause();});
})();
