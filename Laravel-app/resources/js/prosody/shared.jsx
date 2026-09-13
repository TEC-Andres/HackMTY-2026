export function ProsodyHeader() {
    return (
        <header className="w-full flex items-center justify-between text-sm text-slate-600 px-2">
            <a href="/subir_archivo" className="flex items-center gap-2 font-medium text-slate-800">
                <ChameleonIcon className="w-5 h-5 text-indigo-500" />
                Prosody
            </a>
        </header>
    );
}

export function ChameleonIcon({ className = 'w-10 h-10' }) {
    return (
        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-hidden="true">
            <circle cx="24" cy="20" r="14" fill="currentColor" opacity="0.85" />
            <circle cx="19" cy="16" r="2.4" fill="white" />
            <circle cx="29" cy="16" r="2.4" fill="white" />
            <circle cx="19" cy="16" r="1.1" fill="#1e293b" />
            <circle cx="29" cy="16" r="1.1" fill="#1e293b" />
            <path
                d="M30 30c8 2 18 8 18 18a10 10 0 0 1-10 10c-1 0-2-.8-2-2s1-2 2-2a6 6 0 0 0 6-6c0-6-6-11-14-13"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
                fill="none"
            />
        </svg>
    );
}

export function Spinner({ className = 'w-6 h-6 text-indigo-500' }) {
    return (
        <svg className={`${className} animate-spin`} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
    );
}

export function formatPercent(fraction) {
    if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return '—';
    return `${Math.round(fraction * 100)}%`;
}

export function verdictLabel(isSynthetic) {
    if (isSynthetic === null || isSynthetic === undefined) return 'Sin datos';
    return isSynthetic ? 'Sintético' : 'Humano';
}

export function verdictColorClasses(isSynthetic) {
    if (isSynthetic === null || isSynthetic === undefined) {
        return { text: 'text-slate-500', ring: 'ring-slate-200', bg: 'bg-slate-50', bar: 'bg-slate-400' };
    }
    return isSynthetic
        ? { text: 'text-orange-500', ring: 'ring-orange-200', bg: 'bg-orange-50', bar: 'bg-orange-400' }
        : { text: 'text-emerald-500', ring: 'ring-emerald-200', bg: 'bg-emerald-50', bar: 'bg-emerald-400' };
}

export const RESULT_STORAGE_KEY = 'prosody:lastResult';
