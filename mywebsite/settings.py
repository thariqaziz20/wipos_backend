"""
Django settings for mywebsite project.
Deploy-ready for Railway.
"""

from pathlib import Path
import os

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# --- SECURITY ---

# Pakai SECRET_KEY kamu sendiri di sini (boleh pakai yang lama)
SECRET_KEY = 'django-insecure-p!x35pl5suh2-(!&#$amy^3etky2m=#fq2q*(q$hda0$z6ksvp'

# Untuk development / testing boleh True.
# Kalau nanti sudah produksi banget, set False dan pakai env variable.
DEBUG = True

# Railway domain + wildcard
ALLOWED_HOSTS = [
    "*",
    ".up.railway.app",
]

# CSRF untuk domain Railway kamu
CSRF_TRUSTED_ORIGINS = [
    "https://wipos-backend-production.up.railway.app",  # ganti kalau nama subdomain berubah
    "https://*.up.railway.app",
]

# Agar Django tahu kalau request dikirim lewat HTTPS dari proxy Railway
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# --- APPS ---

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "rest_framework",
    "rest_framework.authtoken",
    "django_filters",

    "accounts",
    "background_service",
    "data_wipos",
    "hasil_prediksi",
    "classification",
]

AUTH_USER_MODEL = "accounts.User"

# --- MIDDLEWARE ---

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # untuk static file di Railway
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "mywebsite.urls"

# --- REST FRAMEWORK ---

REST_FRAMEWORK = {
    "NON_FIELD_ERRORS_KEY": "errors",
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        # contoh kalau nanti mau pakai:
        # "rest_framework.permissions.IsAuthenticated",
    ),
}

# --- TEMPLATES ---

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "mywebsite.wsgi.application"

# --- DATABASE ---

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# --- PASSWORD VALIDATION ---

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# --- I18N / TIMEZONE ---

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Jakarta"
USE_I18N = True
USE_TZ = True

# --- STATIC FILES (Railway + WhiteNoise) ---

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"  # tempat collectstatic
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# --- DEFAULT PRIMARY KEY ---

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
