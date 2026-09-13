import { useCallback, useState } from 'react';
import { ChameleonIcon, ProsodyHeader, RESULT_STORAGE_KEY, Spinner, formatPercent, verdictColorClasses, verdictLabel } from './shared';

export default function ProsodyUpload() {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState('idle'); // idle | analyzing | done | error
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files?.[0] ?? null);
        setError(null);
    };

    const handleAnalyze = useCallback(async () => {
        if (!file) return;
        setStatus('analyzing');
        setError(null);

        const formData = new FormData();
        formData.append('audio', file);

        try {
            const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content ?? '';
            const response = await fetch('/subir_archivo', {
                method: 'POST',
                headers: { 'X-CSRF-TOKEN': csrfToken, Accept: 'application/json' },
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message ?? 'No se pudo analizar el archivo.');
            }

            sessionStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(data));
            setResult(data);
            setStatus('done');
        } catch (err) {
            setError(err.message);
            setStatus('error');
        }
    }, [file]);

    const handleReset = () => {
        setFile(null);
        setResult(null);
        setError(null);
        setStatus('idle');
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-emerald-100 flex flex-col items-center px-4 py-8">
            <div className="w-full max-w-md">
                <ProsodyHeader />
            </div>

            <main className="w-full max-w-md mt-10 flex flex-col items-center gap-6">
                {status === 'done' && result ? (
                    <VerdictCard result={result} onReset={handleReset} />
                ) : (
                    <>
                        <h1 className="text-3xl font-semibold text-slate-800">Prosody</h1>
                        <ChameleonIcon className="w-16 h-16 text-indigo-500" />

                        <label
                            htmlFor="audio-input"
                            className="w-full border-2 border-dashed border-indigo-300 rounded-2xl bg-indigo-50/60 hover:bg-indigo-50 transition-colors cursor-pointer flex flex-col items-center justify-center gap-2 py-10 px-6 text-center"
                        >
                            <span className="text-3xl" aria-hidden="true">⬆️</span>
                            <span className="font-medium text-indigo-700">{file ? file.name : 'Subir archivo'}</span>
                            <span className="text-xs text-slate-500">Formato: WAV estéreo, 8kHz</span>
                            <input
                                id="audio-input"
                                type="file"
                                accept=".wav,audio/wav,audio/x-wav"
                                className="hidden"
                                onChange={handleFileChange}
                            />
                        </label>

                        <button
                            type="button"
                            onClick={handleAnalyze}
                            disabled={!file || status === 'analyzing'}
                            className="w-full rounded-full bg-indigo-500 text-white font-medium py-3 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-indigo-600 transition-colors"
                        >
                            {status === 'analyzing' ? 'Analizando…' : 'Analizar'}
                        </button>

                        {status === 'analyzing' && (
                            <div className="flex flex-col items-center gap-2 text-slate-500 text-sm">
                                <Spinner />
                                <p>Analizando: {file?.name}</p>
                                <p>Analizando archivos…</p>
                            </div>
                        )}

                        {status === 'error' && (
                            <p className="text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-lg px-4 py-3 w-full text-center">
                                {error}
                            </p>
                        )}
                    </>
                )}
            </main>
        </div>
    );
}

function VerdictCard({ result, onReset }) {
    const ensemble = result.ensemble ?? {};
    const colors = verdictColorClasses(ensemble.is_synthetic);

    const goToResultados = () => {
        window.location.href = '/subir_archivo/resultados';
    };

    return (
        <div className="w-full flex flex-col items-center gap-6">
            <ChameleonIcon className={`w-14 h-14 ${colors.text}`} />

            <div className="w-full text-center">
                <p className="text-sm text-slate-500 mb-1">Analizando: {result.filename}</p>
                <p className="text-sm text-slate-500">Veredicto:</p>
                <p className={`text-2xl font-semibold ${colors.text}`}>{verdictLabel(ensemble.is_synthetic)}</p>
                <p className="text-sm text-slate-500 mt-3">Confidence:</p>
                <p className={`text-5xl font-bold ${colors.text}`}>{formatPercent(ensemble.confidence)}</p>
            </div>

            <div className={`w-full rounded-2xl ${colors.bg} ring-1 ${colors.ring} p-6 text-center flex flex-col items-center gap-4`}>
                <p className="text-lg font-semibold text-slate-800">Análisis completo</p>
                <button
                    type="button"
                    onClick={goToResultados}
                    className="rounded-full bg-slate-900 text-white text-sm font-medium px-5 py-2 hover:bg-slate-700 transition-colors"
                >
                    Ver resultados detallados
                </button>
            </div>

            <button type="button" onClick={onReset} className="text-sm text-indigo-600 underline">
                Subir otro archivo
            </button>
        </div>
    );
}
