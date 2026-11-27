document.addEventListener('DOMContentLoaded', () => {
  const videos = {
    third: document.getElementById('real-third-player'),
    ego: document.getElementById('real-ego-player')
  };
  const sources = {
    third: document.getElementById('real-third-source'),
    ego: document.getElementById('real-ego-source')
  };
  const playButton = document.getElementById('real-playback-button');
  const timeline = document.getElementById('real-timeline');
  const timeLabel = document.getElementById('real-time');
  const note = document.getElementById('real-sync-note');
  if (Object.values({ ...videos, ...sources, playButton, timeline, timeLabel, note }).some((item) => !item)) return;

  const tasks = {
    button_pressing: 'Button Pressing',
    pepper_picking: 'Pepper Picking',
    rack_cleaning: 'Rack Cleaning'
  };
  const conditions = { stage: 'Stage Occlusion', clean: 'Clean' };
  let selectedTask = 'rack_cleaning';
  let selectedCondition = 'stage';
  let ready = new Set();
  let duration = 0;
  let playing = false;
  let frame = 0;

  function posterPath(task, condition, view) {
    return `static/images/rollouts/${task}_${condition}_${view}.jpg`;
  }

  function formatTime(seconds) {
    const whole = Math.floor(seconds || 0);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
  }

  function updateTime(seconds) {
    const current = Math.min(seconds, duration);
    timeline.value = String(current);
    timeLabel.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
  }

  function leadAndFollower() {
    return videos.third.duration >= videos.ego.duration
      ? [videos.third, videos.ego]
      : [videos.ego, videos.third];
  }

  function pauseBoth() {
    playing = false;
    cancelAnimationFrame(frame);
    Object.values(videos).forEach((video) => video.pause());
    const lead = ready.size === 2 ? leadAndFollower()[0] : null;
    playButton.textContent = lead && lead.currentTime >= duration - 0.05
      ? 'Replay both views'
      : 'Play both views';
  }

  function seekBoth(seconds) {
    for (const video of Object.values(videos)) {
      const target = Math.min(seconds, Math.max(0, video.duration - 0.04));
      video.currentTime = target;
      if (playing && seconds < video.duration - 0.05 && video.paused) {
        video.play().catch(() => pauseBoth());
      }
    }
    updateTime(seconds);
  }

  function followPlayback() {
    if (!playing) return;
    const [lead, follower] = leadAndFollower();
    const current = lead.currentTime;
    if (!follower.ended && current < follower.duration - 0.05 &&
        Math.abs(follower.currentTime - current) > 0.12) {
      follower.currentTime = current;
    }
    updateTime(current);
    if (lead.ended || current >= duration - 0.04) {
      pauseBoth();
      updateTime(duration);
      return;
    }
    frame = requestAnimationFrame(followPlayback);
  }

  async function playBoth() {
    if (ready.size !== 2) return;
    const [lead] = leadAndFollower();
    if (lead.currentTime >= duration - 0.05) seekBoth(0);
    const starts = Object.values(videos)
      .filter((video) => video.currentTime < video.duration - 0.05)
      .map((video) => video.play());
    const results = await Promise.allSettled(starts);
    if (results.some((result) => result.status === 'rejected')) {
      pauseBoth();
      note.textContent = 'Playback could not start. Try the shared play button again.';
      return;
    }
    playing = true;
    playButton.textContent = 'Pause both views';
    note.textContent = 'Both views share one timeline. The shorter view holds its final frame while the other finishes.';
    followPlayback();
  }

  function updateSelection() {
    pauseBoth();
    ready = new Set();
    duration = 0;
    playButton.disabled = true;
    playButton.textContent = 'Loading videos';
    timeline.disabled = true;
    timeline.max = '1';
    updateTime(0);
    const title = tasks[selectedTask];
    const condition = conditions[selectedCondition];

    for (const view of Object.keys(videos)) {
      const video = videos[view];
      sources[view].src = `static/videos/${selectedTask}_${selectedCondition}_${view}.mp4`;
      video.poster = posterPath(selectedTask, selectedCondition, view);
      video.setAttribute('aria-label', `${title} under ${condition}, ${view === 'ego' ? 'active-camera' : 'third-person'} view`);
      video.load();
    }

    document.getElementById('real-rollout-title').textContent = title;
    document.getElementById('real-rollout-description').textContent = `${title} under ${condition}, shown from both cameras.`;
    note.textContent = 'Both views start together. The shorter view holds its final frame while the other finishes.';
    document.querySelectorAll('[data-real-task]').forEach((card) => {
      card.setAttribute('aria-pressed', String(card.dataset.realTask === selectedTask));
      card.querySelector('img').src = posterPath(card.dataset.realTask, selectedCondition, 'third');
    });
    document.querySelectorAll('[data-real-condition]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.realCondition === selectedCondition));
    });
  }

  Object.entries(videos).forEach(([view, video]) => {
    video.addEventListener('loadedmetadata', () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) return;
      ready.add(view);
      if (ready.size !== 2) return;
      duration = Math.max(videos.third.duration, videos.ego.duration);
      timeline.max = String(duration);
      timeline.disabled = false;
      playButton.disabled = false;
      playButton.textContent = 'Play both views';
      updateTime(0);
    });
    video.addEventListener('error', () => {
      pauseBoth();
      playButton.disabled = true;
      timeline.disabled = true;
      note.textContent = 'This video could not be loaded. Try another task or condition.';
    });
  });

  playButton.addEventListener('click', () => {
    if (playing) pauseBoth();
    else playBoth();
  });
  timeline.addEventListener('input', () => seekBoth(Number(timeline.value)));
  document.querySelectorAll('[data-real-task]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedTask = button.dataset.realTask;
      updateSelection();
    });
  });
  document.querySelectorAll('[data-real-condition]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedCondition = button.dataset.realCondition;
      updateSelection();
    });
  });

  updateSelection();
});
