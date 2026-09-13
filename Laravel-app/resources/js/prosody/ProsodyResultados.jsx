import { useEffect, useState } from 'react';
import { ChameleonIcon, ProsodyHeader, RESULT_STORAGE_KEY, formatPercent, verdictColorClasses, verdictLabel } from './shared';

export default function ProsodyResultados() {
    const [result, setResult] = useState(undefined); // undefined = loading, null = missing

    useEffect(() => {
        try {
            const raw = sessionStorage.getItem(RESULT_STORAGE_KEY);
            setResult(raw ? JSON.parse(raw) : null);
        } catch {
            setResult(null);
        }
    }, []);

    if (result === undefined) return null;

    if (!result) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-emerald-100 flex flex-col items-center px-4 py-8">
                <div className="w-full max-w-2xl">
                    <ProsodyHeader />
                </div>
                <main className="w-full max-w-md mt-16 flex flex-col items-center gap-4 text-center">
                    <ChameleonIcon className="w-14 h-14 text-slate-400" />
                    <p className="text-slate-600">Todavía no hay un análisis reciente en esta sesión.</p>
                    <a href="/subir_archivo" className="rounded-full bg-indigo-500 text-white font-medium px-5 py-2 hover:bg-indigo-600 transition-colors">
                        Subir un archivo
                    </a>
                </main>
            </div>
        );
    }

    const ensemble = result.ensemble ?? {};
    const ensembleColors = verdictColorClasses(ensemble.is_synthetic);

    const parametros = [
        { key: 'timing', label: 'Tiempos de distribución', data: result.timing },
        { key: 'lexical', label: 'Análisis léxico', data: result.lexical },
        { key: 'resonance', label: 'Resonancia', data: result.resonance },
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-emerald-100 flex flex-col items-center px-4 py-8">
            <div className="w-full max-w-2xl">
                <ProsodyHeader />
            </div>

            <main className="w-full max-w-2xl mt-8 flex flex-col gap-8">
                <h1 className="text-2xl font-semibold text-slate-800">Resumen Detallado: Prosody</h1>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-stretch">
                    <div className="bg-white/70 rounded-2xl ring-1 ring-slate-200 overflow-hidden">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-100 text-slate-600">
                                <tr>
                                    <th className="px-4 py-2 font-medium">Nombre de archivo</th>
                                    <th className="px-4 py-2 font-medium">Veredicto</th>
                                    <th className="px-4 py-2 font-medium">Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td className="px-4 py-3 text-slate-700 truncate max-w-[10rem]">{result.filename}</td>
                                    <td className={`px-4 py-3 font-medium ${ensembleColors.text}`}>{verdictLabel(ensemble.is_synthetic)}</td>
                                    <td className="px-4 py-3 text-slate-700">{formatPercent(ensemble.confidence)}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div className={`rounded-2xl ${ensembleColors.bg} ring-1 ${ensembleColors.ring} p-6 flex flex-col justify-center`}>
                        <p className="text-sm text-slate-600 mb-1">Analizando: {result.filename}</p>
                        <p className="text-sm text-slate-600 mb-2">Porcentaje de acierto (ensemble)</p>
                        <p className={`text-5xl font-bold ${ensembleColors.text}`}>{formatPercent(ensemble.confidence)}</p>
                    </div>
                </div>

                <section>
                    <h2 className="text-lg font-semibold text-slate-800 mb-1">Desglose de parámetros individuales</h2>
                    <p className="text-sm text-slate-500 mb-4">
                        Nivel de confianza que llevó cada parámetro a la decisión detrás del audio.
                    </p>

                    <div className="flex flex-col gap-3">
                        {parametros.map(({ key, label, data }) => (
                            <ParametroRow key={key} label={label} data={data} />
                        ))}
                    </div>
                </section>

                {result.report && (
                    <section>
                        <h2 className="text-lg font-semibold text-slate-800 mb-2">Reporte completo</h2>
                        <pre className="w-full overflow-x-auto text-xs bg-slate-900 text-slate-100 rounded-2xl p-4 whitespace-pre-wrap">
                            {result.report}
                        </pre>
                    </section>
                )}

                <a
                    href="/subir_archivo"
                    className="self-center rounded-full bg-slate-900 text-white text-sm font-medium px-5 py-2 hover:bg-slate-700 transition-colors"
                >
                    Subir otro archivo
                </a>
            </main>
        </div>
    );
}

function ParametroRow({ label, data }) {
    if (!data) {
        return (
            <div className="rounded-xl bg-white/70 ring-1 ring-slate-200 px-4 py-3 flex items-center justify-between text-slate-400 text-sm">
                <span>{label}</span>
                <span>No disponible para este audio</span>
            </div>
        );
    }

    const colors = verdictColorClasses(data.is_synthetic);
    const widthPct = Math.round((data.confidence ?? 0) * 100);

    return (
        <div className="rounded-xl bg-white/70 ring-1 ring-slate-200 px-4 py-3">
            <div className="flex items-center justify-between text-sm mb-2">
                <span className="font-medium text-slate-700">{label}</span>
                <span className={`font-medium ${colors.text}`}>
                    {verdictLabel(data.is_synthetic)} · {formatPercent(data.confidence)}
                </span>
            </div>
            <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                <div className={`h-full ${colors.bar}`} style={{ width: `${widthPct}%` }} />
            </div>
        </div>
    );
}
