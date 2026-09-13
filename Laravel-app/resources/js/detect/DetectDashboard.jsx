import { useCallback, useEffect, useRef, useState } from 'react';

const POLL_INTERVAL_MS = 2000;

function formatPercent(fraction, decimals = 1) {
    if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return '—';
    return `${(fraction * 100).toFixed(decimals)}%`;
}

function formatSeconds(seconds) {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—';
    return `${seconds.toFixed(3)} s`;
}

function formatNumber(value, decimals = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return value.toFixed(decimals);
}

const STATUS_META = {
    error: { label: 'Error', className: 'text-rose-600 bg-rose-50' },
    skip: { label: 'Sin audio', className: 'text-slate-400 bg-slate-50' },
};

const METRIC_ROWS = [
    { key: 'accuracy', label: 'Accuracy', format: (s) => formatPercent(s.accuracy) },
    { key: 'balanced_accuracy', label: 'Balanced Accuracy', format: (s) => formatPercent(s.balanced_accuracy) },
    { key: 'auc', label: 'AUC', format: (s) => formatPercent(s.auc) },
    { key: 'tpr_synthetic', label: 'Synthetic TPR', format: (s) => formatPercent(s.tpr_synthetic) },
    { key: 'tnr_human', label: 'Human TNR', format: (s) => formatPercent(s.tnr_human) },
    { key: 'mean_latency_s', label: 'Mean Latency', format: (s) => formatSeconds(s.mean_latency_s) },
    { key: 'max_latency_s', label: 'Max Latency', format: (s) => formatSeconds(s.max_latency_s) },
    { key: 'brier', label: 'Brier Score', format: (s) => formatNumber(s.brier) },
    { key: 'errors', label: 'Errors', format: (s) => `${s.errors}` },
];

export default function DetectDashboard() {
    const [report, setReport] = useState(undefined); // undefined = loading, null = no report yet (404)
    const [fetchError, setFetchError] = useState(null);
    const pollRef = useRef(null);

    const fetchReport = useCallback(async () => {
        try {
            const res = await fetch('/detect/data', { headers: { Accept: 'application/json' } });
            if (res.status === 404) {
                setReport(null);
                setFetchError(null);
                return;
            }
            const data = await res.json();
            if (!res.ok) {
                setFetchError(data?.message ?? `Error del servidor (${res.status}).`);
                return;
            }
            setReport(data);
            setFetchError(null);
        } catch {
            setFetchError('No se pudo conectar con el servidor.');
        }
    }, []);

    useEffect(() => {
        fetchReport();
        pollRef.current = setInterval(fetchReport, POLL_INTERVAL_MS);
        return () => clearInterval(pollRef.current);
    }, [fetchReport]);

    const isLoading = report === undefined && !fetchError;
    const hasReport = Boolean(report) && typeof report === 'object';
    const isRunning = hasReport && !report.summary;

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-emerald-100 px-4 py-8">
            <div className="max-w-5xl mx-auto flex flex-col gap-6">
                <header>
                    <h1 className="text-2xl font-semibold text-slate-800">Detect — Evaluación por lotes</h1>
                    <p className="text-sm text-slate-500">
                        Resultados en vivo de{' '}
                        <code className="bg-slate-100 px-1 rounded">scripts/check_endpoint.py</code> contra{' '}
                        <code className="bg-slate-100 px-1 rounded">POST /detect</code>.
                    </p>
                </header>

                {isLoading && <p className="text-slate-500">Cargando…</p>}

                {fetchError && (
                    <p className="text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-lg px-4 py-3">
                        {fetchError}
                    </p>
                )}

                {!isLoading && !hasReport && !fetchError && <EmptyState />}

                {hasReport && (
                    <>
                        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
                            <span>
                                URL: <code className="bg-slate-100 px-1 rounded">{report.url}</code>
                            </span>
                            <span>·</span>
                            <span>
                                {report.results.length}/{report.total} llamadas procesadas
                            </span>
                            {isRunning && (
                                <span className="flex items-center gap-1 text-indigo-600 font-medium">
                                    <Spinner /> Ejecutándose…
                                </span>
                            )}
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6 items-start">
                            <CallsTable results={report.results} />
                            <SummaryPanel total={report.total} summary={report.summary} isRunning={isRunning} />
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

function EmptyState() {
    return (
        <div className="rounded-2xl bg-white/70 ring-1 ring-slate-200 p-6 flex flex-col gap-3">
            <p className="text-slate-700 font-medium">Todavía no hay una evaluación reciente.</p>
            <p className="text-sm text-slate-500">
                Corre esto en tu terminal (desde <code className="bg-slate-100 px-1 rounded">fastapi/</code>, con el
                servidor ya corriendo):
            </p>
            <pre className="text-xs bg-slate-900 text-slate-100 rounded-xl p-4 overflow-x-auto">
{`python scripts/check_endpoint.py --url http://127.0.0.1:8001/detect \\
    --audio-dir ../hackmty26/audio --n 50`}
            </pre>
            <p className="text-sm text-slate-500">Esta página se actualiza sola mientras corre.</p>
        </div>
    );
}

function Spinner() {
    return (
        <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
    );
}

function CallsTable({ results }) {
    return (
        <div className="bg-white/70 rounded-2xl ring-1 ring-slate-200 overflow-hidden">
            <div className="max-h-[32rem] overflow-y-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-slate-100 text-slate-600 sticky top-0">
                        <tr>
                            <th className="px-4 py-2 font-medium">Llamada</th>
                            <th className="px-4 py-2 font-medium">Real</th>
                            <th className="px-4 py-2 font-medium">Veredicto</th>
                            <th className="px-4 py-2 font-medium">Confidence</th>
                            <th className="px-4 py-2 font-medium">Latencia</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results.map((row, idx) => (
                            <CallRow key={`${row.anon_id}-${idx}`} row={row} />
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function CallRow({ row }) {
    if (row.status !== 'ok') {
        const meta = STATUS_META[row.status] ?? { label: row.status, className: 'text-slate-500 bg-slate-50' };
        return (
            <tr className="border-t border-slate-100">
                <td className="px-4 py-2 text-slate-700 truncate max-w-[10rem]">{row.anon_id}</td>
                <td className="px-4 py-2 text-slate-500">{row.truth_label ?? '—'}</td>
                <td className="px-4 py-2" colSpan={2}>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${meta.className}`}>
                        {meta.label}
                    </span>
                    {row.error && <span className="ml-2 text-xs text-slate-400 truncate">{row.error}</span>}
                </td>
                <td className="px-4 py-2 text-slate-500">
                    {row.latency_s !== undefined ? formatSeconds(row.latency_s) : '—'}
                </td>
            </tr>
        );
    }

    const verdictColor = row.is_synthetic ? 'text-orange-500' : 'text-emerald-500';
    const correctBadge = row.correct ? 'text-emerald-600 bg-emerald-50' : 'text-rose-600 bg-rose-50';

    return (
        <tr className="border-t border-slate-100">
            <td className="px-4 py-2 text-slate-700 truncate max-w-[10rem]">{row.anon_id}</td>
            <td className="px-4 py-2 text-slate-500">{row.truth_label}</td>
            <td className={`px-4 py-2 font-medium ${verdictColor}`}>
                {row.is_synthetic ? 'Sintético' : 'Humano'}{' '}
                <span className={`ml-1 text-xs px-1.5 py-0.5 rounded-full ${correctBadge}`}>
                    {row.correct ? 'ok' : 'miss'}
                </span>
            </td>
            <td className="px-4 py-2 text-slate-700">{formatPercent(row.confidence, 0)}</td>
            <td className="px-4 py-2 text-slate-500">{formatSeconds(row.latency_s)}</td>
        </tr>
    );
}

function SummaryPanel({ total, summary, isRunning }) {
    return (
        <div className="rounded-2xl bg-white/70 ring-1 ring-slate-200 p-6 flex flex-col gap-3">
            <p className="text-lg font-semibold text-slate-800">Evaluation — {total} calls</p>

            {!summary ? (
                <p className="text-sm text-slate-500">{isRunning ? 'Ejecutando…' : 'Sin resumen todavía.'}</p>
            ) : (
                <dl className="flex flex-col gap-1.5 font-mono text-sm">
                    {METRIC_ROWS.map(({ key, label, format }) => (
                        <div key={key} className="flex items-baseline justify-between gap-4">
                            <dt className="text-slate-500">{label}</dt>
                            <dd className="text-slate-800 font-medium">{format(summary)}</dd>
                        </div>
                    ))}
                </dl>
            )}
        </div>
    );
}
