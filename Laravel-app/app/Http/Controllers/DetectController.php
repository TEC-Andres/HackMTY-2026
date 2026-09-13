<?php

namespace App\Http\Controllers;

use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class DetectController extends Controller
{
    /**
     * GET /detect — live dashboard for FastAPI's POST /detect ensemble.
     */
    public function show()
    {
        return view('detect.index');
    }

    /**
     * GET /detect/data — proxies FastAPI's GET /detect/report (the JSON
     * mirror of whatever scripts/check_endpoint.py last wrote while
     * exercising POST /detect) so the browser can poll it without knowing
     * FASTAPI_BASE_URL. Purely a read of that report; never calls /detect.
     */
    public function data(): JsonResponse
    {
        $baseUrl = rtrim(config('services.fastapi.base_url'), '/');
        $timeout = (int) config('services.fastapi.timeout', 120);

        try {
            $response = Http::timeout($timeout)->acceptJson()->get("{$baseUrl}/detect/report");
        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::warning('Detect report: no se pudo conectar con FastAPI', ['error' => $e->getMessage()]);

            return response()->json([
                'message' => 'No se pudo conectar con el servicio de análisis (FastAPI). '
                    ."Verifica que esté corriendo en {$baseUrl}.",
            ], 502);
        }

        if ($response->status() === 404) {
            return response()->json([
                'message' => $response->json('detail', 'Todavía no hay una evaluación reciente.'),
            ], 404);
        }

        if ($response->failed()) {
            return response()->json([
                'message' => $response->json('detail', 'El servicio de análisis no pudo devolver el reporte.'),
            ], $response->status());
        }

        return response()->json($response->json());
    }
}
