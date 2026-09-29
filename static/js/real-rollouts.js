document.addEventListener('DOMContentLoaded', () => {
  const player = document.getElementById('real-rollout-player');
  const source = document.getElementById('real-rollout-source');
  if (!player || !source) return;

  const tasks = {
    button_pressing: {
      title: 'Button Pressing'
    },
    pepper_picking: {
      title: 'Pepper Picking'
    },
    rack_cleaning: {
      title: 'Rack Cleaning'
    }
  };

  const conditions = {
    occlusion: {
      label: 'with occlusion',
      description: 'A physical robot rollout with an externally introduced occlusion.'
    },
    clean: {
      label: 'without occlusion',
      description: 'A physical robot rollout without an external occluder.'
    }
  };

  let selectedTask = 'rack_cleaning';
  let selectedCondition = 'occlusion';

  function posterPath(task, condition) {
    const path = `static/images/rollouts/real_${task}_${condition}.jpg`;
    return task === 'pepper_picking' && condition === 'occlusion' ? `${path}?v=14s` : path;
  }

  function updateSelection() {
    const task = tasks[selectedTask];
    const condition = conditions[selectedCondition];
    const basename = `real_${selectedTask}_${selectedCondition}`;
    const video = selectedCondition === 'clean' ? `${basename}_web.mp4` : `${basename}.mp4`;

    player.pause();
    source.src = `static/videos/${video}`;
    player.poster = posterPath(selectedTask, selectedCondition);
    player.setAttribute('aria-label', `${task.title} physical robot rollout ${condition.label}`);
    player.load();

    document.getElementById('real-rollout-title').textContent = task.title;
    document.getElementById('real-rollout-description').textContent = condition.description;

    document.querySelectorAll('[data-real-task]').forEach((card) => {
      card.setAttribute('aria-pressed', String(card.dataset.realTask === selectedTask));
      card.querySelector('img').src = posterPath(card.dataset.realTask, selectedCondition);
    });
    document.querySelectorAll('[data-real-condition]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.realCondition === selectedCondition));
    });
  }

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
});
