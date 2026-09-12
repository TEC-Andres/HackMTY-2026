<?php


// Google Verification 
namespace App\Http\Controllers\Auth;

use App\Http\Controllers\Controller;
use App\Models\User;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Laravel\Socialite\Facades\Socialite;

class SocialiteController extends Controller
{
    public function redirectToGoogle()
    {
        return Socialite::driver('google')->redirect();
    }

    //HAck MTY-2026
  public function handleGoogleCallback()
    {
        $googleUser = Socialite::driver('google')->user();
        $email = $googleUser->getEmail();
        $allowedDomain = 'tec.mx'; // Change to your desired domain

        // Check if email ends with the allowed domain
        // Matches @tec.mx, @itesm.mx, or any subdomains like @gda.tec.mx
        if (!preg_match('/@(.*\.)?(tec\.mx|itesm\.mx)$/i', $email)) {
            return redirect()->route('login')->withErrors([
                'email' => "Access denied. Only Tec de Monterrey accounts are allowed.",
            ]);
        }

        $user = User::updateOrCreate(
            ['email' => $email],
            [
                'name' => $googleUser->getName(),
                'google_id' => $googleUser->getId(),
                'password' => Hash::make(uniqid()),
            ]
        );

        Auth::login($user);

        return redirect()->intended('/dashboard');
    }
}

/*
namespace App\Http\Controllers\Auth;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;

class SocialiteController extends Controller
{
    //
}
*/