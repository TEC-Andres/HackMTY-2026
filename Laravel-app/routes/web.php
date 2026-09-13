<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\ProsodyController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('welcome');
});

// Prosody: subir un archivo de audio individual y analizarlo (HackMTY-2026)
Route::get('/subir_archivo', [ProsodyController::class, 'showUpload'])->name('prosody.upload');
Route::post('/subir_archivo', [ProsodyController::class, 'analyze'])->name('prosody.analyze');
Route::get('/subir_archivo/resultados', [ProsodyController::class, 'showResultados'])->name('prosody.resultados');

Route::get('/dashboard', function () {
    return view('dashboard');
})->middleware(['auth', 'verified'])->name('dashboard');

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

require __DIR__.'/auth.php';

// Google OAuth routes (HackMTY-2026)
use App\Http\Controllers\Auth\SocialiteController;

Route::get('/auth/google', [SocialiteController::class, 'redirectToGoogle'])->name('google.login');
Route::get('/auth/google/callback', [SocialiteController::class, 'handleGoogleCallback']);

//Routes 2, fake google
// Temporary test route to simulate a Google login response
Route::get('/test-google-login', function () {
    // Change this email to test pass/fail (@tec.mx vs @gmail.com)
    $email = 'A01234567@tec.mx'; 

    if (!preg_match('/@(.*\.)?(tec\.mx|itesm\.mx)$/i', $email)) {
        return redirect()->route('login')->withErrors([
            'email' => "Access denied. Only Tec de Monterrey accounts are allowed.",
        ]);
    }

    $user = \App\Models\User::updateOrCreate(
        ['email' => $email],
        [
            'name' => 'Test Student',
            'google_id' => 'fake-google-id-123',
            'password' => \Illuminate\Support\Facades\Hash::make(uniqid()),
        ]
    );

    \Illuminate\Support\Facades\Auth::login($user);

    return redirect()->intended('/dashboard');
});