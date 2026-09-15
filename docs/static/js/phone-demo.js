'use strict';
document.addEventListener('DOMContentLoaded', async () => {
  // User-initiated listening only; never autoplay these research examples.
  document.addEventListener('play', event => {
    if (event.target.tagName === 'AUDIO') {
      document.querySelectorAll('audio').forEach(player => {
        if (player !== event.target) player.pause();
      });
    }
  }, true);
  try {
    const response = await fetch('static/data/phone-examples.json');
    if (!response.ok) throw new Error('Phone example inventory could not be loaded');
    const examples = await response.json();
    const base = 'static/assets/';
    document.querySelectorAll('.phone-room').forEach(card => {
      const room = Number(card.dataset.roomId);
      const mode = card.querySelector('.clap-mode');
      const repetition = card.querySelector('.clap-repeat');
      const caption = card.querySelector('.example-caption');
      const update = () => {
        const matches = examples.filter(e => e.room_id === room && e.mode === mode.value);
        const chosen = repetition.value === 'selected'
          ? matches.find(e => e.selected_for_demo)
          : matches.find(e => e.repetition === Number(repetition.value));
        if (!chosen) throw new Error('Requested recording is missing');
        for (const [kind, key] of [['recorded','recorded_audio'],['inferred','inferred_audio']]) {
          const player = card.querySelector('.' + kind + '-audio');
          player.pause();
          player.src = base + chosen[key];
          player.setAttribute('aria-label', chosen.room + ', ' + chosen.mode
            + ', repetition ' + chosen.repetition + ', ' + (kind === 'recorded' ? 'recorded clap' : 'inferred RIR'));
          card.querySelector('.download-' + kind).href = base + chosen[key];
        }
        let text = chosen.mode + ' · repetition ' + chosen.repetition;
        if (chosen.selected_for_paper) text += ' · paper-selected example';
        else if (chosen.selected_for_demo) text += ' · input-selected example';
        if (chosen.source_clipped) text += ' · original recording contains clipping';
        if (chosen.valid_input_duration_s < 1) text += ' · recording support ' + chosen.valid_input_duration_s.toFixed(3) + ' s';
        caption.textContent = text;
        card.dataset.eventIndex = chosen.event_index;
        card.dataset.repetition = chosen.repetition;
      };
      mode.addEventListener('change', update);
      repetition.addEventListener('change', update);
      update();
    });
    document.documentElement.dataset.phoneDemoReady = 'true';
  } catch (error) {
    console.error(error);
    document.documentElement.dataset.phoneDemoReady = 'error';
  }
});
