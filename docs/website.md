# Website maintenance

The project page is `index.html`. It includes the title, authors, affiliations, abstract, four research figures, an interactive simulation rollout browser, and a physical robot video demo.

The public paper is available at https://arxiv.org/abs/2609.37292 and the code repository is https://github.com/HCPLab-SYSU/BAVO-Bench.

The page includes the paper's arXiv BibTeX entry with a copy button.

## Add figures

The source PDFs and their web-ready PNGs are in `static/images/`:

- `Introduction.pdf` → `intro.png` — overview figure
- `benchmark.pdf` → `benchmark.png` — benchmark figure
- `method.pdf` → `method.png` — A-FAR method figure
- `deploy.pdf` → `deploy.png` — physical deployment sequence

The PNGs are displayed on the page with the supplied figure captions. The PDFs are retained as source material; the paper PDF is not needed.

## Add rollout videos

The 15 simulation MP4s are in `static/videos/`, named by task and condition. They are already H.264 and show the third-person scene on the left and active-camera view on the right. The page switches between five tasks and three visibility conditions using `static/js/rollouts.js`; it loads one video at a time.

Six physical robot MP4s cover Button Pressing, Pepper Picking, and Rack Cleaning under Clean and Occlusion. They are displayed below the Physical Deployment figure. The task cards and rollout buttons switch one shared player using `static/js/real-rollouts.js`; the page does not label these videos as Stage or Random-time experiments. Poster frames for both video sections are in `static/images/rollouts/`.

The supplied Clean MP4s used HEVC. Their `_clean_web.mp4` derivatives use H.264 for broader browser support; the original HEVC files are not included in this website repository. The Occlusion MP4s are already H.264.
