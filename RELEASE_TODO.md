# Release follow-up

- Add a single evaluation entry point for the paper protocol: five tasks × three occlusion conditions, 20 episodes per condition, fixed reset seeds and episode step limits. Report per-task success rate and TPS, random-time CVG, and the five-task macro average. For CVG, use a 60-step window at 25 Hz and average valid events within each episode before averaging episodes. This is intentionally deferred from the current repository cleanup.
