# PSL Face Analyzer (MVP)

A compact, responsive React + Vite frontend based on the uploaded Stitch export. It combines the four exported screen variants into a single page with two states:

- **Upload state:** choose/drag a JPG, PNG, or WebP portrait.
- **Results state:** run MediaPipe Face Landmarker in the browser and show a landmark overlay plus simple geometry indicators.

## Important limitations

- The PSL-style value is a **prototype heuristic**, not an official PSL evaluation and not a scientifically validated attractiveness measure. Its formula is visible in `src/main.jsx` in `measureFace()` and is included for experimentation only.
- 2D facial landmarks are sensitive to head angle, expression, camera distance, lens distortion, lighting, and image orientation.
- This does not assess skin quality, health, personality, or a person's worth.
- The program does not upload the selected photo or store it in an app database. It uses a local `blob:` URL for preview and processing in the current browser session. Resetting removes the selected photo from the app state.
- The MediaPipe WASM runtime and model are downloaded from public CDNs when first used. MediaPipe's official notice says its input image is processed on-device and is not sent to Google, while the library may send usage/performance metrics. See https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md. For strict offline operation, the runtime and model should be self-hosted in a later iteration.

## Requirements

- Node.js (LTS recommended)
- npm
- Modern browser, e.g. Chrome or Edge
- Internet access for initial dependency installation and MediaPipe runtime/model download

## Run locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`). Select a clear, front-facing portrait and click **Analyze face**. No `npm run` command is needed beyond the scripts above.

## Production build

```bash
npm run build
npm run preview
```

The output will be generated in `dist/`.

## GitHub Pages

`vite.config.js` uses relative asset paths (`base: './'`) to support project-site paths. To publish the Vite build, configure a GitHub Actions workflow that runs `npm ci`, `npm run build`, and publishes the `dist/` folder. Follow the current Vite guide: https://vite.dev/guide/static-deploy.

## Structure

```text
psl-face-analyzer/
├── index.html
├── package.json
├── vite.config.js
├── src/
│   ├── main.jsx
│   └── styles.css
└── README.md
```
