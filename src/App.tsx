import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Braces,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Eye,
  FileJson,
  FlaskConical,
  Gauge,
  Grid3X3,
  Layers3,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
  ServerCog,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import './App.css'

type Scheme = 'cfg' | 'fitted' | 'interval'
type ViewName = 'lab' | 'atlas' | 'build'
type RenderMode = 'retina' | 'raw'

interface SchemeSample {
  image: string
  mask: string
  sha256: string
  saturation: number
  contrast: number
  laplacianEnergy: number
  latentNormMax: number
  clipScore: number
}

interface DemoSample {
  promptIndex: number
  prompt: string
  seed: number
  schemes: Record<Scheme, SchemeSample>
}

interface FinalManifest {
  source: {
    model: string
    scheduler: string
    guidanceScale: number
    w: number
    numSteps: number
    sourceTreeSha256: string
    selection: string
  }
  samples: DemoSample[]
}

interface TraceStep {
  step: number
  timestep: number
  sigma: number
  sigmaNext: number
  r: number
  h: number | null
  coefficient: number
  latentNorm: number
  saturation: number
  image: string
  mask: string
}

interface TraceManifest {
  model: string
  scheduler: string
  prompt: string
  seed: number
  guidanceScale: number
  w: number
  numSteps: number
  sharedInitial: { image: string; mask: string; saturation: number }
  schemes: Record<Scheme, TraceStep[]>
}

interface RetinaAsset {
  display: string
  sourceWidth: number
  sourceHeight: number
  displayWidth: number
  displayHeight: number
  sourceSha256: string
  displaySha256: string
}

interface RetinaManifest {
  scope: string
  processor: { scale: number; resample: string; sharpen: string; format: string; quality: number }
  assets: Record<string, RetinaAsset>
}

interface SdxlScheme {
  image: string
  mask: string
  sha256: string
  saturation: number
  coefficientMin: number
  coefficientFinal: number
  latentNormMax: number
}

interface SdxlManifest {
  scope: string
  selection: string
  model: string
  scheduler: string
  prompt: string
  seed: number
  guidanceScale: number
  w: number
  numSteps: number
  height: number
  width: number
  schemes: Record<Scheme, SdxlScheme>
}

interface EvidenceScheme {
  fid: number
  kid: number
  ampP95: number
  clipP95: number
  targetAccuracy: number
  targetConfidence: number
  terminalCoefficient: number
}

interface CifarCell {
  w: number
  n: number
  schemes: Record<Scheme, EvidenceScheme>
}

interface LatentCell {
  id: string
  label: string
  summary: Record<Scheme, Record<string, number | string>>
  bootstrap: Record<string, Record<string, number | string | boolean>>
  shutdownContrast: Record<string, number | string>
}

interface EvidenceManifest {
  cifar: { dataset: string; samplesPerCell: number; source: string; cells: CifarCell[] }
  latentDiffusion: { dataset: string; model: string; samplesPerCell: number; cells: LatentCell[] }
  scope: string[]
}

const schemeMeta: Record<Scheme, { label: string; short: string; detail: string }> = {
  cfg: { label: 'Vanilla CFG', short: 'CFG', detail: 'Euler coefficient' },
  fitted: { label: 'Fitted CFG', short: 'Fitted', detail: 'Layer-fitted coefficient' },
  interval: { label: 'Terminal interval', short: 'Interval', detail: 'Guidance shutdown' },
}

const base = import.meta.env.BASE_URL
const assetUrl = (path: string) => `${base}${path.replace(/^\//, '')}`
const renderedAsset = (path: string, retina: RetinaManifest, mode: RenderMode) => (
  mode === 'retina' ? retina.assets[path]?.display ?? path : path
)
const format = (value: number, digits = 3) => value.toFixed(digits)
const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`

function useJson<T>(path: string): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    fetch(assetUrl(path))
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
        return response.json() as Promise<T>
      })
      .then((payload) => active && setData(payload))
      .catch((reason: Error) => active && setError(reason.message))
    return () => {
      active = false
    }
  }, [path])
  return { data, error }
}

function MethodMark({ scheme }: { scheme: Scheme }) {
  return <span className={`method-mark method-${scheme}`} aria-hidden="true" />
}

function ImagePanel({
  scheme,
  image,
  mask,
  xray,
  saturation,
  clipScore,
  coefficient,
  selected,
  retina,
  renderMode,
}: {
  scheme: Scheme
  image: string
  mask: string
  xray: boolean
  saturation: number
  clipScore: number | null
  coefficient: number | null
  selected: boolean
  retina: RetinaManifest
  renderMode: RenderMode
}) {
  const meta = schemeMeta[scheme]
  const renderedImage = renderedAsset(image, retina, renderMode)
  return (
    <figure className={`comparison-panel ${selected ? 'is-mobile-selected' : ''}`} data-scheme={scheme}>
      <div className="image-stage">
        <img
          src={assetUrl(renderedImage)}
          alt={`${meta.label} output`}
          data-render-mode={renderMode}
          draggable={false}
        />
        {xray && <img className="mask-layer" src={assetUrl(mask)} alt="Clipped-pixel overlay" draggable={false} />}
        <div className="method-label">
          <MethodMark scheme={scheme} />
          <span>{meta.label}</span>
        </div>
        {xray && (
          <div className="xray-key">
            <ScanLine size={14} /> clipped pixels
          </div>
        )}
      </div>
      <figcaption>
        <span>
          <small>Saturation</small>
          <strong>{percent(saturation, 2)}</strong>
        </span>
        <span>
          <small>{coefficient === null ? 'CLIP score' : 'Step coefficient'}</small>
          <strong>{coefficient === null ? format(clipScore ?? 0) : format(coefficient)}</strong>
        </span>
      </figcaption>
    </figure>
  )
}

function CoefficientChart({ trace, step }: { trace: TraceManifest; step: number }) {
  const width = 760
  const height = 184
  const pad = { left: 44, right: 18, top: 20, bottom: 30 }
  const series = (['cfg', 'fitted'] as Scheme[]).map((scheme) => ({ scheme, values: trace.schemes[scheme] }))
  const values = series.flatMap((entry) => entry.values.map((item) => item.coefficient))
  const min = Math.min(-1, ...values)
  const max = Math.max(0.4, ...values)
  const x = (index: number) => pad.left + ((width - pad.left - pad.right) * index) / Math.max(1, trace.numSteps - 1)
  const y = (value: number) => pad.top + ((height - pad.top - pad.bottom) * (max - value)) / (max - min)
  const selectedIndex = Math.max(0, Math.min(trace.numSteps - 1, step - 1))
  return (
    <div className="coefficient-chart" aria-label="CFG and fitted guidance coefficients by denoising step">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <line className="chart-axis" x1={pad.left} x2={width - pad.right} y1={y(0)} y2={y(0)} />
        {[1, 4, 8, 12].map((tick) => (
          <g key={tick}>
            <line className="chart-tick" x1={x(tick - 1)} x2={x(tick - 1)} y1={pad.top} y2={height - pad.bottom} />
            <text x={x(tick - 1)} y={height - 9} textAnchor="middle">{tick}</text>
          </g>
        ))}
        {series.map(({ scheme, values: points }) => (
          <polyline
            key={scheme}
            className={`chart-line chart-${scheme}`}
            points={points.map((item, index) => `${x(index)},${y(item.coefficient)}`).join(' ')}
          />
        ))}
        {step > 0 && (
          <>
            <line className="chart-cursor" x1={x(selectedIndex)} x2={x(selectedIndex)} y1={pad.top} y2={height - pad.bottom} />
            {series.map(({ scheme, values: points }) => (
              <circle
                key={scheme}
                className={`chart-point chart-${scheme}`}
                cx={x(selectedIndex)}
                cy={y(points[selectedIndex].coefficient)}
                r="5"
              />
            ))}
          </>
        )}
        <text className="axis-label" x={8} y={18}>c</text>
        <text className="axis-label" x={width - 36} y={height - 9}>step</text>
      </svg>
      <div className="chart-legend" aria-hidden="true">
        <span><MethodMark scheme="cfg" />w(r - 1)</span>
        <span><MethodMark scheme="fitted" />r^(1+w) - r</span>
      </div>
    </div>
  )
}

function NativeHero({ manifest }: { manifest: SdxlManifest }) {
  const [selected, setSelected] = useState<Scheme>('fitted')
  return (
    <section className="native-hero" aria-labelledby="native-hero-title">
      <div className="native-hero-heading">
        <div>
          <span className="eyebrow">Native {manifest.width} · Stable Diffusion XL</span>
          <h1 id="native-hero-title">{manifest.prompt}</h1>
        </div>
        <div className="native-hero-facts" aria-label="SDXL run configuration">
          <span><small>Guidance</small><strong>{manifest.guidanceScale}</strong></span>
          <span><small>NFE</small><strong>{manifest.numSteps}</strong></span>
          <span><small>Seed</small><strong>{manifest.seed}</strong></span>
        </div>
      </div>

      <div className="native-hero-switch segmented" aria-label="Visible SDXL method">
        {(Object.keys(schemeMeta) as Scheme[]).map((scheme) => (
          <button key={scheme} type="button" aria-pressed={selected === scheme} onClick={() => setSelected(scheme)}>
            <MethodMark scheme={scheme} />{schemeMeta[scheme].short}
          </button>
        ))}
      </div>

      <div className="native-hero-grid">
        {(Object.keys(schemeMeta) as Scheme[]).map((scheme) => {
          const entry = manifest.schemes[scheme]
          return (
            <figure
              key={scheme}
              className={`native-output ${selected === scheme ? 'is-selected' : ''}`}
              data-scheme={scheme}
            >
              <div className="native-image-stage">
                <img src={assetUrl(entry.image)} alt={`${schemeMeta[scheme].label} SDXL output`} draggable={false} />
                <div className="method-label"><MethodMark scheme={scheme} />{schemeMeta[scheme].label}</div>
              </div>
              <figcaption>
                <span><small>Native pixels</small><strong>{manifest.width}²</strong></span>
                <span><small>Saturation</small><strong>{percent(entry.saturation, 2)}</strong></span>
              </figcaption>
            </figure>
          )
        })}
      </div>
      <p className="native-hero-scope">{manifest.scope} Same prompt, seed, network, schedule, and NFE.</p>
    </section>
  )
}

function LabView({
  finalManifest,
  trace,
  retina,
  sdxl,
}: {
  finalManifest: FinalManifest
  trace: TraceManifest | null
  retina: RetinaManifest
  sdxl: SdxlManifest | null
}) {
  const [selectedIndex, setSelectedIndex] = useState(finalManifest.samples[0].promptIndex)
  const [step, setStep] = useState(trace?.numSteps ?? finalManifest.source.numSteps)
  const [playing, setPlaying] = useState(false)
  const [xray, setXray] = useState(false)
  const [mobileScheme, setMobileScheme] = useState<Scheme>('fitted')
  const [renderMode, setRenderMode] = useState<RenderMode>('retina')
  const sample = finalManifest.samples.find((entry) => entry.promptIndex === selectedIndex) ?? finalManifest.samples[0]
  const traceActive = Boolean(trace && sample.seed === trace.seed)
  const maxStep = trace?.numSteps ?? finalManifest.source.numSteps

  useEffect(() => {
    if (!playing || !traceActive) return
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current >= maxStep) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 620)
    return () => window.clearInterval(timer)
  }, [playing, traceActive, maxStep])

  useEffect(() => {
    if (!traceActive) {
      setPlaying(false)
      setStep(maxStep)
    }
  }, [traceActive, maxStep])

  const frameFor = (scheme: Scheme) => {
    const final = sample.schemes[scheme]
    if (!trace || !traceActive) {
      return { image: final.image, mask: final.mask, saturation: final.saturation, coefficient: null, clipScore: final.clipScore }
    }
    if (step === 0) {
      return {
        image: trace.sharedInitial.image,
        mask: trace.sharedInitial.mask,
        saturation: trace.sharedInitial.saturation,
        coefficient: null,
        clipScore: null,
      }
    }
    const frame = trace.schemes[scheme][step - 1]
    return { ...frame, clipScore: step === maxStep ? final.clipScore : null }
  }

  const current = trace && step > 0 ? trace.schemes.cfg[step - 1] : null
  const play = () => {
    if (!traceActive) return
    if (step >= maxStep) setStep(0)
    setPlaying(true)
  }

  return (
    <div className="view lab-view">
      {sdxl && <NativeHero manifest={sdxl} />}

      <section className="lab-toolbar" aria-label="Demo controls">
        <label className="field prompt-field">
          <span>Matched prompt</span>
          <select
            value={selectedIndex}
            onChange={(event) => setSelectedIndex(Number(event.target.value))}
          >
            {finalManifest.samples.map((entry) => (
              <option key={entry.promptIndex} value={entry.promptIndex}>{entry.prompt}</option>
            ))}
          </select>
        </label>
        <div className="run-facts" aria-label="Run configuration">
          <span><small>Guidance</small><strong>{finalManifest.source.guidanceScale}</strong></span>
          <span><small>NFE</small><strong>{finalManifest.source.numSteps}</strong></span>
          <span><small>Seed</small><strong>{sample.seed}</strong></span>
        </div>
        <div className="resolution-switch segmented" aria-label="Image resolution">
          <button
            type="button"
            aria-pressed={renderMode === 'retina'}
            title="Deterministic 2x display derivative"
            onClick={() => setRenderMode('retina')}
          >
            Retina
          </button>
          <button
            type="button"
            aria-pressed={renderMode === 'raw'}
            title="Exact 512 experiment output"
            onClick={() => setRenderMode('raw')}
          >
            Raw 512
          </button>
        </div>
        <button className={`tool-button ${xray ? 'is-active' : ''}`} type="button" aria-pressed={xray} onClick={() => setXray(!xray)}>
          {xray ? <Eye size={18} /> : <ScanLine size={18} />}
          X-ray
        </button>
      </section>

      <div className="prompt-line">
        <span className="eyebrow">Stable Diffusion 1.5 · deterministic DDIM</span>
        <h1>{sample.prompt}</h1>
        <p>Same prompt, seed, and network evaluations. Only the guidance coefficient changes.</p>
      </div>

      <div className="mobile-scheme-switch segmented" aria-label="Visible method">
        {(Object.keys(schemeMeta) as Scheme[]).map((scheme) => (
          <button key={scheme} type="button" aria-pressed={mobileScheme === scheme} onClick={() => setMobileScheme(scheme)}>
            <MethodMark scheme={scheme} />{schemeMeta[scheme].short}
          </button>
        ))}
      </div>

      <section className="comparison-grid" aria-label="Matched method comparison">
        {(Object.keys(schemeMeta) as Scheme[]).map((scheme) => {
          const frame = frameFor(scheme)
          return (
            <ImagePanel
              key={`${sample.promptIndex}-${scheme}-${step}`}
              scheme={scheme}
              image={frame.image}
              mask={frame.mask}
              xray={xray}
              saturation={frame.saturation}
              clipScore={frame.clipScore}
              coefficient={frame.coefficient ?? null}
              selected={mobileScheme === scheme}
              retina={retina}
              renderMode={renderMode}
            />
          )
        })}
      </section>

      <section className={`trace-console ${traceActive ? '' : 'is-inactive'}`}>
        <div className="timeline-controls">
          <button className="icon-button" type="button" title={playing ? 'Pause trace' : 'Play trace'} onClick={() => playing ? setPlaying(false) : play()} disabled={!traceActive}>
            {playing ? <Pause size={19} /> : <Play size={19} />}
          </button>
          <button className="icon-button" type="button" title="Previous step" onClick={() => setStep(Math.max(0, step - 1))} disabled={!traceActive || step === 0}>
            <ChevronLeft size={19} />
          </button>
          <input
            aria-label="Denoising step"
            type="range"
            min="0"
            max={maxStep}
            value={step}
            onChange={(event) => {
              setPlaying(false)
              setStep(Number(event.target.value))
            }}
            disabled={!traceActive}
          />
          <button className="icon-button" type="button" title="Next step" onClick={() => setStep(Math.min(maxStep, step + 1))} disabled={!traceActive || step === maxStep}>
            <ChevronRight size={19} />
          </button>
          <button className="icon-button" type="button" title="Replay trace" onClick={() => { setStep(0); setPlaying(true) }} disabled={!traceActive}>
            <RotateCcw size={18} />
          </button>
          <strong>Step {step}/{maxStep}</strong>
        </div>
        {!traceActive && <p className="trace-note">Select the ski prompt to replay the exact per-step trajectory. Other examples show fixed final outputs.</p>}
        {trace && traceActive && (
          <div className="trace-layout">
            <CoefficientChart trace={trace} step={step} />
            <div className="step-readout">
              <span><small>sigma</small><strong>{current ? format(current.sigma, 2) : 'initial'}</strong></span>
              <span><small>r</small><strong>{current ? format(current.r, 3) : '—'}</strong></span>
              <span><small>h</small><strong>{current?.h == null ? (step === 0 ? '—' : '∞') : format(current.h, 3)}</strong></span>
              <span><small>CFG c</small><strong>{current ? format(trace.schemes.cfg[step - 1].coefficient) : '—'}</strong></span>
              <span><small>Fitted c</small><strong>{current ? format(trace.schemes.fitted[step - 1].coefficient) : '—'}</strong></span>
            </div>
          </div>
        )}
      </section>

      <section className="formula-band">
        <div>
          <span className="eyebrow">One-coefficient repair</span>
          <h2>The model and network calls stay fixed.</h2>
        </div>
        <code><span className="removed">w(r - 1)</span><ArrowRight size={20} /><span className="added">r^(1+w) - r</span></code>
      </section>

      <p className="display-provenance">
        Retina mode uses the same deterministic 2x display transform for every method. Raw 512 remains the experiment source.
      </p>

      <section className="example-strip" aria-labelledby="examples-heading">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Matched outputs</span>
            <h2 id="examples-heading">More fixed examples</h2>
          </div>
          <p>Left half CFG, right half fitted CFG.</p>
        </div>
        <div className="sample-tiles">
          {finalManifest.samples.map((entry) => (
            <button
              key={entry.promptIndex}
              type="button"
              className={entry.promptIndex === selectedIndex ? 'is-selected' : ''}
              onClick={() => setSelectedIndex(entry.promptIndex)}
              aria-label={`View prompt: ${entry.prompt}`}
            >
              <span className="split-thumb">
                <img className="left" src={assetUrl(entry.schemes.cfg.image)} alt="" />
                <img className="right" src={assetUrl(entry.schemes.fitted.image)} alt="" />
                <span className="split-rule" />
              </span>
              <span>{entry.prompt}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}

function MetricTable({ cell }: { cell: CifarCell }) {
  const metrics: Array<[string, (value: EvidenceScheme) => string]> = [
    ['FID ↓', (value) => format(value.fid, 2)],
    ['KID ↓', (value) => format(value.kid, 4)],
    ['Residual amp p95 ↓', (value) => format(value.ampP95, 2)],
    ['Clipping p95 ↓', (value) => percent(value.clipP95, 1)],
    ['Target accuracy ↑', (value) => percent(value.targetAccuracy, 1)],
  ]
  return (
    <div className="metric-table-wrap">
      <table className="metric-table">
        <thead>
          <tr><th>Metric</th>{(['cfg', 'fitted', 'interval'] as Scheme[]).map((scheme) => <th key={scheme}><MethodMark scheme={scheme} />{schemeMeta[scheme].short}</th>)}</tr>
        </thead>
        <tbody>
          {metrics.map(([label, get]) => (
            <tr key={label}>
              <th>{label}</th>
              {(['cfg', 'fitted', 'interval'] as Scheme[]).map((scheme) => <td key={scheme}>{get(cell.schemes[scheme])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AtlasView({ evidence }: { evidence: EvidenceManifest }) {
  const [selectedKey, setSelectedKey] = useState('8-8')
  const cells = evidence.cifar.cells
  const selected = cells.find((cell) => `${cell.w}-${cell.n}` === selectedKey) ?? cells[0]
  const byKey = useMemo(() => new Map(cells.map((cell) => [`${cell.w}-${cell.n}`, cell])), [cells])
  const ws = [4, 6.5, 8]
  const ns = [8, 16, 32]
  return (
    <div className="view atlas-view">
      <section className="page-intro">
        <span className="eyebrow">Failure atlas</span>
        <h1>Where the one-line repair matters.</h1>
        <p>Each cell is a 5k balanced-class CIFAR-10 EDM comparison at matched NFE. Click a cell to inspect all three schemes.</p>
      </section>

      <section className="atlas-layout">
        <div className="heatmap" aria-label="CIFAR fitted CFG experiment grid">
          <div className="heatmap-corner">w \ N</div>
          {ns.map((n) => <div className="heatmap-header" key={n}>{n}</div>)}
          {ws.flatMap((w) => [
            <div className="heatmap-row-label" key={`${w}-label`}>{w}</div>,
            ...ns.map((n) => {
              const cell = byKey.get(`${w}-${n}`)!
              const fidGain = cell.schemes.cfg.fid - cell.schemes.fitted.fid
              const tailGain = cell.schemes.cfg.ampP95 / cell.schemes.fitted.ampP95
              const strength = fidGain > 10 ? 'high' : fidGain > 4 ? 'medium' : 'low'
              return (
                <button
                  key={`${w}-${n}`}
                  type="button"
                  className={`heatmap-cell strength-${strength} ${selectedKey === `${w}-${n}` ? 'is-selected' : ''}`}
                  onClick={() => setSelectedKey(`${w}-${n}`)}
                  aria-label={`w ${w}, N ${n}: FID improves by ${fidGain.toFixed(2)}, residual tail ${tailGain.toFixed(1)} times lower`}
                >
                  <strong>−{format(fidGain, 1)} FID</strong>
                  <span>{format(tailGain, 1)}× tail</span>
                </button>
              )
            }),
          ])}
        </div>
        <div className="cell-detail">
          <div className="detail-title">
            <span className="eyebrow">Selected cell</span>
            <h2>w = {selected.w}, N = {selected.n}</h2>
          </div>
          <MetricTable cell={selected} />
          <p className="interpretation">
            Fitted CFG keeps the guidance signal while reducing the terminal tail. Interval guidance can score lower FID at larger N, but its class accuracy is lower in every grid cell.
          </p>
        </div>
      </section>

      <section className="ldm-evidence">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Cross-domain transfer</span>
            <h2>Stable Diffusion 1.5</h2>
          </div>
          <p>5k fixed COCO captions per cell.</p>
        </div>
        <div className="ldm-rows">
          {evidence.latentDiffusion.cells.map((cell) => {
            const bootstrap = cell.bootstrap.fitted
            const ratio = Number(bootstrap.sat_p95_ratio_med)
            const clipDelta = Number(bootstrap.clip_mean_delta_med)
            const intervalDelta = Number(cell.shutdownContrast.clip_mean_delta_med)
            return (
              <article key={cell.id} className="ldm-row">
                <div>
                  <strong>{cell.label}</strong>
                  <span>{(100 * (1 - ratio)).toFixed(1)}% lower saturation p95</span>
                </div>
                <div className="bar-track" aria-label={`${(100 * (1 - ratio)).toFixed(1)} percent lower saturation p95`}>
                  <span style={{ width: `${Math.max(0, Math.min(100, 100 * (1 - ratio)))}%` }} />
                </div>
                <div className="ldm-numbers">
                  <span><small>CLIP vs CFG</small>{clipDelta > 0 ? '+' : ''}{format(clipDelta, 4)}</span>
                  <span><small>CLIP vs interval</small>+{format(intervalDelta, 4)}</span>
                </div>
              </article>
            )
          })}
        </div>
      </section>

      <section className="scope-band">
        <ShieldCheck size={26} />
        <div>
          <h2>Scoped claim</h2>
          <p>{evidence.scope[0]} {evidence.scope[1]}</p>
        </div>
      </section>
    </div>
  )
}

function BuildView({
  finalManifest,
  evidence,
  retina,
}: {
  finalManifest: FinalManifest
  evidence: EvidenceManifest
  retina: RetinaManifest
}) {
  return (
    <div className="view build-view">
      <section className="page-intro">
        <span className="eyebrow">System view</span>
        <h1>From theorem to inspectable software.</h1>
        <p>The public experience is fast and deterministic; the same artifact contract can be produced by a live GPU worker.</p>
      </section>

      <section className="architecture" aria-label="Application architecture">
        <div className="architecture-node">
          <Layers3 size={24} />
          <strong>React lab</strong>
          <span>Timeline, X-ray, atlas</span>
        </div>
        <ArrowRight className="architecture-arrow" />
        <div className="architecture-node">
          <FileJson size={24} />
          <strong>Trace contract</strong>
          <span>WebP frames + JSON manifests</span>
        </div>
        <ArrowRight className="architecture-arrow" />
        <div className="architecture-node">
          <ServerCog size={24} />
          <strong>Hyak GPU worker</strong>
          <span>PyTorch, Diffusers, DDIM</span>
        </div>
      </section>

      <section className="implementation-grid">
        <div className="code-panel">
          <div className="panel-heading"><Braces size={19} /><span>Sampler change</span></div>
          <pre><code><span className="diff-remove">- c = w * (r - 1.0)</span>{'\n'}<span className="diff-add">+ c = r ** (1.0 + w) - r</span>{'\n'}{'  y_next = D_c + r * (y - D_c) + c * (D_u - D_c)'}</code></pre>
        </div>
        <div className="reliability-list">
          <div className="panel-heading"><ShieldCheck size={19} /><span>Reliability</span></div>
          {[
            'Coefficient identities and one-jump self-test',
            'Matched prompt, seed, network, and NFE',
            'SHA256 for every public final image',
            'Display derivatives separated from raw evidence',
            'Responsive desktop and mobile visual checks',
            'Static fallback independent of GPU availability',
          ].map((item) => <span key={item}><Check size={16} />{item}</span>)}
        </div>
      </section>

      <section className="provenance-section">
        <div className="section-heading">
          <div><span className="eyebrow">Reproducibility</span><h2>Artifact provenance</h2></div>
          <a className="text-button" href={assetUrl('/demo/final/manifest.json')}><FileJson size={17} />Manifest</a>
        </div>
        <dl className="provenance-grid">
          <div><dt>Model</dt><dd>{finalManifest.source.model}</dd></div>
          <div><dt>Scheduler</dt><dd>{finalManifest.source.scheduler}</dd></div>
          <div><dt>Cell</dt><dd>g={finalManifest.source.guidanceScale}, N={finalManifest.source.numSteps}</dd></div>
          <div><dt>Evidence</dt><dd>{evidence.cifar.samplesPerCell.toLocaleString()} images × 9 CIFAR cells</dd></div>
          <div><dt>Display layer</dt><dd>{Object.keys(retina.assets).length} assets · {retina.processor.scale}× deterministic</dd></div>
          <div className="hash"><dt>Source snapshot</dt><dd>{finalManifest.source.sourceTreeSha256}</dd></div>
        </dl>
      </section>

      <section className="scope-list">
        <div><CircleStop size={23} /><h2>What this demo does not claim</h2></div>
        {evidence.scope.map((item) => <p key={item}><X size={15} />{item}</p>)}
      </section>
    </div>
  )
}

function App() {
  const [view, setView] = useState<ViewName>('lab')
  const final = useJson<FinalManifest>('/demo/final/manifest.json')
  const trace = useJson<TraceManifest>('/demo/trace/manifest.json')
  const retina = useJson<RetinaManifest>('/demo/retina/manifest.json')
  const sdxl = useJson<SdxlManifest>('/demo/sdxl/manifest.json')
  const evidence = useJson<EvidenceManifest>('/demo/evidence.json')
  const ready = final.data && retina.data && evidence.data

  return (
    <div className="app-shell">
      <header className="app-header">
        <button className="brand" type="button" onClick={() => setView('lab')} aria-label="Open Guidance Stability Lab">
          <span className="brand-mark"><Activity size={21} /></span>
          <span><strong>Guidance Stability Lab</strong><small>Fitted CFG</small></span>
        </button>
        <nav className="primary-nav" aria-label="Primary navigation">
          <button type="button" aria-current={view === 'lab' ? 'page' : undefined} onClick={() => setView('lab')}><FlaskConical size={17} />Lab</button>
          <button type="button" aria-current={view === 'atlas' ? 'page' : undefined} onClick={() => setView('atlas')}><Grid3X3 size={17} />Atlas</button>
          <button type="button" aria-current={view === 'build' ? 'page' : undefined} onClick={() => setView('build')}><Braces size={17} />Build</button>
        </nav>
        <div className="header-status"><span />Exact matched trace</div>
      </header>

      <main>
        {!ready && (
          <div className="loading-state">
            <Gauge size={28} />
            <strong>Loading experiment bundle</strong>
            <span>{final.error || retina.error || evidence.error || 'Reading image and metric manifests…'}</span>
          </div>
        )}
        {ready && view === 'lab' && (
          <LabView finalManifest={final.data!} trace={trace.data} retina={retina.data!} sdxl={sdxl.data} />
        )}
        {ready && view === 'atlas' && <AtlasView evidence={evidence.data!} />}
        {ready && view === 'build' && <BuildView finalManifest={final.data!} evidence={evidence.data!} retina={retina.data!} />}
      </main>

      <footer>
        <span><Sparkles size={15} />Same prompt. Same seed. Same NFE. One coefficient changed.</span>
        <span>Research engineering portfolio · 2026</span>
      </footer>
    </div>
  )
}

export default App
