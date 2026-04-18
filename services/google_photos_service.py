"""
Google Photos service for downloading and managing photos from Google Photos library.

Authentication: OAuth2 (user account) - requires browser redirect
OAuth2 does NOT support service accounts - only user accounts can access personal photos.

OAuth2 Setup:
  1. Google Cloud Console → APIs & Services → Enable Photos Library API
  2. Create OAuth2 credentials (Client ID + Secret)
  3. Register redirect URIs: http://localhost:8085 (no trailing slash!)
  4. Run with --client-id and --client-secret, or use --creds-file

OAuth Fix (2026-04-13):
  google-auth-oauthlib's run_local_server() appends a trailing slash to the
  redirect URI it sends to Google (e.g. http://localhost:8085/).
  Google Cloud Console does NOT allow registering URIs with trailing slashes,
  so the two never match → Error 400: redirect_uri_mismatch.

  Fix: pass redirect_uri_override to strip the trailing slash, and ensure
  the client_config redirect_uris list has no trailing slashes either.
"""
import logging
import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

PHOTOS_SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
MEDIA_METADATA_FIELDS = 'mediaItems(id,mediaMetadata,filename,productUrl,mimeType,baseUrl),nextPageToken'

# Ports to try — register ALL of these in Google Cloud Console WITHOUT trailing slash:
#   http://localhost:8085
#   http://localhost:8086
#   http://localhost:8087
#   http://localhost:8088
OAUTH_FALLBACK_PORTS = [8085, 8086, 8087, 8088]


class GooglePhotosService:
    """Service for interacting with Google Photos API."""

    def __init__(
        self,
        credentials_file: str = None,
        client_id: str = None,
        client_secret: str = None,
        token_file: str = '.google_photos_token.json',
        output_dir: str = 'D:\\Photos',
    ):
        self.credentials_file = credentials_file
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_file = token_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.service = None
        self.credentials = None
        self._authenticate()

    # ── OAuth ─────────────────────────────────────────────────────────────────

    def _try_run_local_server(self, flow: InstalledAppFlow) -> Credentials:
        """
        Run the OAuth local-server flow, trying each port in OAUTH_FALLBACK_PORTS.

        KEY FIX: google-auth-oauthlib appends '/' to the redirect_uri it sends,
        producing e.g. 'http://localhost:8085/' which Google rejects because the
        console only stores 'http://localhost:8085' (no trailing slash).

        We work around this by monkey-patching the flow's redirect_uri after
        run_local_server() has set it up but before the browser redirect, using
        the redirect_uri_override parameter available in newer versions, or by
        patching the flow object directly.
        """
        last_error = None

        for port in OAUTH_FALLBACK_PORTS:
            try:
                # Tell the library the exact URI — no trailing slash.
                # run_local_server() accepts redirect_uri_trailing_slash=False
                # in google-auth-oauthlib >= 0.5.0
                logger.info(f"Starting OAuth server on http://localhost:{port}")
                logger.info(
                    f"  Make sure http://localhost:{port} (NO trailing slash) "
                    f"is in Google Cloud Console → OAuth client → Authorized redirect URIs"
                )

                try:
                    # Preferred: newer API parameter
                    credentials = flow.run_local_server(
                        port=port,
                        prompt='consent',
                        open_browser=True,
                        redirect_uri_trailing_slash=False,
                        success_message=(
                            'Authentication complete! You can close this tab.'
                        ),
                    )
                except TypeError:
                    # Older oauthlib version — patch manually
                    flow.redirect_uri = f'http://localhost:{port}'
                    credentials = flow.run_local_server(
                        port=port,
                        prompt='consent',
                        open_browser=True,
                        success_message=(
                            'Authentication complete! You can close this tab.'
                        ),
                    )

                logger.info(f"OAuth completed successfully on port {port}")
                return credentials

            except OSError as e:
                logger.warning(f"Port {port} unavailable: {e}, trying next…")
                last_error = e
            except Exception:
                raise  # Non-port error (e.g. user cancelled) — propagate

        raise RuntimeError(
            f"Could not start OAuth server on any of {OAUTH_FALLBACK_PORTS}. "
            f"Last error: {last_error}. Free one of these ports and retry."
        )

    def _authenticate(self) -> None:
        """Authenticate with Google Photos API using OAuth2."""
        logger.info("Authenticating with Google Photos API (OAuth2 only)")

        # Load existing token
        if os.path.exists(self.token_file):
            try:
                self.credentials = Credentials.from_authorized_user_file(
                    self.token_file, PHOTOS_SCOPES
                )
                logger.info(f"Loaded existing credentials from {self.token_file}")
            except Exception as e:
                logger.warning(f"Failed to load token: {e}. Will re-authenticate.")
                self.credentials = None

        # Refresh or create
        if not self.credentials or not self.credentials.valid:
            if (
                self.credentials
                and self.credentials.expired
                and self.credentials.refresh_token
            ):
                try:
                    self.credentials.refresh(Request())
                    logger.info("Refreshed expired credentials")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    self.credentials = None

            if not self.credentials:
                # Build the flow
                if self.credentials_file and os.path.exists(self.credentials_file):
                    logger.info(f"Using credentials file: {self.credentials_file}")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, PHOTOS_SCOPES
                    )
                elif self.client_id and self.client_secret:
                    logger.info("Using client_id / client_secret")
                    # IMPORTANT: No trailing slashes here — must match Console exactly.
                    client_config = {
                        "installed": {
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": [
                                f"http://localhost:{p}"          # ← NO trailing slash
                                for p in OAUTH_FALLBACK_PORTS
                            ] + ["urn:ietf:wg:oauth:2.0:oob"],
                        }
                    }
                    flow = InstalledAppFlow.from_client_config(client_config, PHOTOS_SCOPES)
                else:
                    raise ValueError(
                        "Must provide either credentials_file, or both client_id and client_secret.\n\n"
                        "Option 1 - OAuth2 with credentials file:\n"
                        "  GooglePhotosService(credentials_file='path/to/oauth-credentials.json')\n\n"
                        "Option 2 - OAuth2 with client ID/secret:\n"
                        "  GooglePhotosService(client_id='YOUR_ID', client_secret='YOUR_SECRET')\n"
                        "  Must register redirect URIs in Google Cloud Console (NO trailing slashes):\n"
                        "    http://localhost:8085\n"
                        "    http://localhost:8086\n"
                        "    http://localhost:8087\n"
                        "    http://localhost:8088\n"
                        "\n"
                        "NOTE: Service accounts are NOT supported by Google Photos Library API.\n"
                        "      Only OAuth2 with user accounts works."
                    )

                self.credentials = self._try_run_local_server(flow)
                logger.info("Completed OAuth2 authentication")

        # Save credentials
        with open(self.token_file, 'w') as f:
            f.write(self.credentials.to_json())

        # Build service
        self._build_service()

    def _build_service(self) -> None:
        """Build the Google Photos API service."""
        self.service = build(
            'photoslibrary', 'v1',
            credentials=self.credentials,
            discoveryServiceUrl=(
                'https://photoslibrary.googleapis.com/$discovery/rest?version=v1'
            ),
        )
        logger.info("Google Photos API service initialized")

    # ── List & download ───────────────────────────────────────────────────────

    def list_photos(self, page_size: int = 100) -> List[Dict]:
        """List all photos in the user's Google Photos library."""
        all_photos = []
        next_page_token = None

        while True:
            logger.info(f"Fetching photos (page size: {page_size})")
            params = {'pageSize': page_size, 'fields': MEDIA_METADATA_FIELDS}
            if next_page_token:
                params['pageToken'] = next_page_token

            results = self.service.mediaItems().list(**params).execute()
            media_items = results.get('mediaItems', [])
            all_photos.extend(media_items)
            logger.info(f"  Retrieved {len(media_items)} photos (total: {len(all_photos)})")

            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break

        logger.info(f"Total photos in library: {len(all_photos)}")
        return all_photos

    def download_photo(
        self,
        photo: Dict,
        preserve_metadata: bool = True,
    ) -> Optional[Tuple[Path, Dict]]:
        """Download a single photo."""
        try:
            photo_id  = photo.get('id')
            filename  = photo.get('filename', f'photo_{photo_id}.jpg')
            base_url  = photo.get('baseUrl')
            media_metadata = photo.get('mediaMetadata', {})

            if not base_url:
                logger.warning(f"No download URL for photo {photo_id}")
                return None

            mime_type    = photo.get('mimeType', '')
            suffix       = '=dv' if mime_type.startswith('video/') else '=d'
            download_url = f"{base_url}{suffix}"

            output_path = self.output_dir / filename
            counter = 1
            stem = output_path.stem
            ext  = output_path.suffix
            while output_path.exists():
                output_path = self.output_dir / f"{stem}_{counter}{ext}"
                counter += 1

            logger.info(f"Downloading: {filename}")
            response = requests.get(download_url, timeout=60, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Saved: {output_path.name} ({size_mb:.1f} MB)")

            if preserve_metadata:
                meta_file = output_path.with_name(output_path.stem + '_metadata.json')
                meta = {
                    'photo_id':     photo_id,
                    'filename':     filename,
                    'media_type':   mime_type,
                    'created_time': media_metadata.get('creationTime'),
                    'width':        media_metadata.get('width'),
                    'height':       media_metadata.get('height'),
                    'camera_make':  media_metadata.get('photo', {}).get('cameraMake'),
                    'camera_model': media_metadata.get('photo', {}).get('cameraModel'),
                }
                with open(meta_file, 'w') as f:
                    json.dump(meta, f, indent=2)

            return output_path, photo

        except Exception as e:
            logger.error(f"Failed to download photo {photo.get('id')}: {e}")
            return None

    def download_all_photos(
        self,
        max_photos: Optional[int] = None,
        batch_size: int = 10,
        preserve_metadata: bool = True,
        order_by: str = "creationTime",
    ) -> Dict:
        """Download all photos from Google Photos library."""
        logger.info("Fetching photo list from Google Photos…")
        photos = self.list_photos()

        if order_by == "creationTime":
            photos.sort(
                key=lambda p: p.get('mediaMetadata', {}).get('creationTime', ''),
                reverse=False,
            )
            logger.info("Sorted by creation time (oldest first)")

        if max_photos:
            photos = photos[:max_photos]
            logger.info(f"Limited to {max_photos} photos")

        stats = {
            'total': len(photos), 'successful': 0, 'failed': 0, 'skipped': 0,
            'total_size_mb': 0, 'start_time': datetime.now(), 'downloaded_files': [],
        }
        logger.info(f"Starting download of {stats['total']} photos to {self.output_dir}…")

        for idx, photo in enumerate(photos, 1):
            if idx % batch_size == 0 or idx == 1:
                logger.info(f"Progress: {idx}/{stats['total']}")

            mime_type = photo.get('mimeType', '')
            if not mime_type.startswith('image/') and not mime_type.startswith('video/'):
                stats['skipped'] += 1
                continue

            result = self.download_photo(photo, preserve_metadata)
            if result:
                file_path, _ = result
                stats['successful'] += 1
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                stats['downloaded_files'].append(str(file_path))
            else:
                stats['failed'] += 1

        stats['end_time'] = datetime.now()
        stats['duration_seconds'] = (
            stats['end_time'] - stats['start_time']
        ).total_seconds()

        logger.info("=" * 60)
        logger.info("Download Complete!")
        logger.info(f"  Downloaded: {stats['successful']}")
        logger.info(f"  Failed:     {stats['failed']}")
        logger.info(f"  Skipped:    {stats['skipped']}")
        logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")
        logger.info(f"  Duration:   {stats['duration_seconds']:.1f}s")
        logger.info("=" * 60)

        return stats
