document.addEventListener('DOMContentLoaded', () => {
  const player = document.getElementById('rollout-player');
  const source = document.getElementById('rollout-source');
  if (!player || !source) return;

  const tasks = {
    cube_transfer_plate: {
      title: 'Cube Handoff',
      summary: 'Handoff a cube between arms and place it on the target plate.'
    },
    block_stack: {
      title: 'Block Stacking',
      summary: 'Move the base block into position and stack the top block on it.'
    },
    drawer_storage: {
      title: 'Drawer Stowing',
      summary: 'Open a drawer, place an object inside, and close it.'
    },
    slide_pickup: {
      title: 'Slide Retrieval',
      summary: 'Slide an object to expose its handle, then retrieve it.'
    },
    dustpan_sweep: {
      title: 'Dustpan Sweeping',
      summary: 'Position the dustpan and sweep the target object into it.'
    }
  };

  const conditions = {
    clean: {
      title: 'Clean',
      note: 'Clean shows the task without a benchmark occluder.'
    },
    stage: {
      title: 'Stage Occlusion',
      note: 'Stage Occlusion introduces task-relevant occlusions at predefined task transitions.'
    },
    random: {
      title: 'Random-time Occlusion',
      note: 'Random-time Occlusion introduces visibility disruptions at sampled times during task execution.'
    }
  };

  let selectedTask = 'cube_transfer_plate';
  let selectedCondition = 'random';

  function updateSelection() {
    const task = tasks[selectedTask];
    const condition = conditions[selectedCondition];
    const filename = `${selectedTask}_${selectedCondition}`;

    player.pause();
    source.src = `static/videos/${filename}.mp4`;
    player.poster = `static/images/rollouts/${filename}.jpg`;
    player.setAttribute('aria-label', `${task.title} under ${condition.title}, with third-person and active-camera views`);
    player.load();

    document.getElementById('rollout-title').textContent = task.title;
    document.getElementById('rollout-summary').textContent = task.summary;
    document.getElementById('rollout-condition-note').textContent = condition.note;

    document.querySelectorAll('[data-rollout-task]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.rolloutTask === selectedTask));
    });
    document.querySelectorAll('[data-rollout-condition]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.rolloutCondition === selectedCondition));
    });
  }

  document.querySelectorAll('[data-rollout-task]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedTask = button.dataset.rolloutTask;
      updateSelection();
    });
  });

  document.querySelectorAll('[data-rollout-condition]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedCondition = button.dataset.rolloutCondition;
      updateSelection();
    });
  });
});
