

import Alpine from 'alpinejs';

window.Alpine = Alpine;

Alpine.start();

const uploadRoot = document.getElementById('prosody-upload-root');
if (uploadRoot) {
    Promise.all([import('react'), import('react-dom/client'), import('./prosody/ProsodyUpload')]).then(
        ([{ default: React }, { createRoot }, { default: ProsodyUpload }]) => {
            createRoot(uploadRoot).render(React.createElement(ProsodyUpload));
        },
    );
}

const resultadosRoot = document.getElementById('prosody-resultados-root');
if (resultadosRoot) {
    Promise.all([import('react'), import('react-dom/client'), import('./prosody/ProsodyResultados')]).then(
        ([{ default: React }, { createRoot }, { default: ProsodyResultados }]) => {
            createRoot(resultadosRoot).render(React.createElement(ProsodyResultados));
        },
    );
}

const detectRoot = document.getElementById('detect-root');
if (detectRoot) {
    Promise.all([import('react'), import('react-dom/client'), import('./detect/DetectDashboard')]).then(
        ([{ default: React }, { createRoot }, { default: DetectDashboard }]) => {
            createRoot(detectRoot).render(React.createElement(DetectDashboard));
        },
    );
}
