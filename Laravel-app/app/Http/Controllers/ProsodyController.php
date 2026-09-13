<?php

namespace App\Http\Controllers;

use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Validation\ValidationException;

class ProsodyController extends Controller
{
    /**
     * GET /subir_archivo — upload screen (Prosody).
     */
    public function showUpload()
    {
        return view('prosody.subir-archivo');
    }

    /**
     * GET /subir_archivo/resultados — detailed parameter breakdown screen.
     */
    public function showResultados()
    {
        return view('prosody.resultados');
    }

    /**
     * POST /subir_archivo — uploads a single WAV file to the FastAPI
     * detection service (POST /detect/STTLexicalAnalysis?verbose=true) and
     * returns the ensemble verdict plus the per-parameter breakdown
     * (timing, lexical, resonance) used by the resultados screen.
     */
    public function analyze(Request $request): JsonResponse
    {
        try {
            $validated = $request->validate([
                'audio' => ['required', 'file', 'max:20480', 'mimes:wav'],
                'channel' => ['nullable', 'integer', 'min:0'],
            ]);
        } catch (ValidationException $e) {
            return response()->json([
                'message' => 'Sube un archivo de audio WAV válido.',
                'errors' => $e->errors(),
            ], 422);
        }

        $file = $request->file('audio');
        $audioBase64 = base64_encode($file->get());
        $channel = (int) ($validated['channel'] ?? 0);

        $baseUrl = rtrim(config('services.fastapi.base_url'), '/');
        $timeout = (int) config('services.fastapi.timeout', 120);

        try {
            $response = Http::timeout($timeout)
                ->acceptJson()
                ->post("{$baseUrl}/detect/STTLexicalAnalysis?verbose=true", [
                    'audio_base64' => $audioBase64,
                    'channel' => $channel,
                ]);
        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::warning('Prosody: no se pudo conectar con FastAPI', ['error' => $e->getMessage()]);

            return response()->json([
                'message' => 'No se pudo conectar con el servicio de análisis (FastAPI). '
                    ."Verifica que esté corriendo en {$baseUrl}.",
            ], 502);
        }

        if ($response->failed()) {
            return response()->json([
                'message' => $response->json('detail', 'El servicio de análisis rechazó el archivo.'),
            ], $response->status());
        }

        $result = $response->json();

        return response()->json([
            'filename' => $file->getClientOriginalName(),
            'ensemble' => $result['ensemble'] ?? null,
            'timing' => $result['timing'] ?? null,
            'lexical' => $result['lexical'] ?? null,
            'resonance' => $result['resonance'] ?? null,
            'agreement' => $result['agreement'] ?? null,
            'report' => $result['report'] ?? null,
        ]);
    }
}
